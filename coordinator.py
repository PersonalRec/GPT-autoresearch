"""
Collaborative autoresearch coordinator.

Bridges the autoresearch experiment loop with the Ensue memory network,
enabling SETI@home-style distributed research across multiple GPU participants.

Uses `requests` (already in pyproject.toml) for JSON-RPC calls. Zero new deps.

Usage:
    from coordinator import Coordinator
    coord = Coordinator()  # reads ENSUE_API_KEY or .autoresearch-key
    coord.join_hub()
    coord.claim_experiment("a7f3b2", {"LR": 0.04}, "increase LR to 0.04")
    coord.publish_result("a7f3b2", result_dict, open("train.py").read())
"""

import base64
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from typing import Any, Optional

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HUB_ORG = "sai_autoresearch-community"
API_URL = "https://api.ensue-network.ai/"
KEY_FILE = ".autoresearch-key"

CLAIM_TTL = 900              # 15 min soft expiry (3x expected 5-min experiment)
VERIFY_DELAY = 2             # seconds between claim and verify
SEMANTIC_THRESHOLD = 0.92    # block if active claim is this similar
MAX_CLAIM_ATTEMPTS = 5       # alternatives before giving up
SYNC_EVERY_N = 5             # pull global best every N experiments

# ---------------------------------------------------------------------------
# Base JSON-RPC
# ---------------------------------------------------------------------------

def _get_api_key() -> Optional[str]:
    """Read API key from env var or key file."""
    key = os.environ.get("ENSUE_API_KEY")
    if key:
        return key.strip()
    if os.path.exists(KEY_FILE):
        with open(KEY_FILE) as f:
            return f.read().strip()
    return None


def ensue_rpc(api_key: str, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Make a JSON-RPC call to the Ensue MCP API."""
    payload = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {"name": tool_name, "arguments": arguments},
        "id": 1,
    }
    resp = requests.post(
        API_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=30,
    )
    resp.raise_for_status()

    # Response may have SSE "data: " prefix
    text = resp.text.strip()
    if text.startswith("data: "):
        text = text[len("data: "):]

    data = json.loads(text)

    if "error" in data:
        raise RuntimeError(f"RPC error: {data['error']}")

    # Extract text content from result
    result = data.get("result", {})
    content = result.get("content", [])
    if content and isinstance(content, list):
        first = content[0]
        if isinstance(first, dict) and "text" in first:
            return json.loads(first["text"])
    return result


def _experiment_hash(description: str) -> str:
    """Hash an experiment description for dedup keying."""
    return hashlib.sha256(description.lower().strip().encode()).hexdigest()[:12]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_remote_url() -> Optional[str]:
    """Get the GitHub HTTPS URL for the current repo."""
    try:
        url = subprocess.check_output(
            ["git", "remote", "get-url", "origin"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        # Convert SSH to HTTPS: git@github.com:user/repo.git -> https://github.com/user/repo
        if url.startswith("git@github.com:"):
            url = "https://github.com/" + url[len("git@github.com:"):]
        if url.endswith(".git"):
            url = url[:-4]
        return url
    except Exception:
        return None


def _git_branch() -> Optional[str]:
    """Get the current git branch name."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def _git_commit_short() -> Optional[str]:
    """Get the short commit hash of HEAD."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------

class Coordinator:
    """
    Synchronous coordinator for collaborative autoresearch.

    All methods catch exceptions and return gracefully so the training loop
    never crashes due to network issues.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or _get_api_key()
        self.agent_id: Optional[str] = None
        self.experiment_count = 0

    @property
    def connected(self) -> bool:
        return self.api_key is not None

    def _rpc(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """RPC call with the stored API key."""
        if not self.api_key:
            raise RuntimeError("No API key configured")
        return ensue_rpc(self.api_key, tool_name, arguments)

    # --- Onboarding ---

    def join_hub(self, invite_token: str) -> dict[str, Any]:
        """Claim the hub invite to join autoresearch-community."""
        try:
            result = self._rpc("claim_invite", {"token": invite_token})
            print(f"[coordinator] Joined hub: {result}")
            return result
        except Exception as e:
            print(f"[coordinator] join_hub failed: {e}")
            return {"error": str(e)}

    def test_connectivity(self) -> bool:
        """Test if the API key works."""
        try:
            self._rpc("list_keys", {"limit": 1})
            return True
        except Exception:
            return False

    # --- Work Claiming ---

    def check_claimed(self, experiment_hash: str) -> bool:
        """Check if an experiment is already claimed (active) or completed."""
        try:
            # Check if result already exists
            result_key = f"@{HUB_ORG}/results/{experiment_hash}"
            result = self._rpc("get_memory", {"key_names": [result_key]})
            results = result.get("results", [])
            if results and results[0].get("status") == "success":
                return True  # already done

            # Check for active claim
            claim_key = f"@{HUB_ORG}/claims/{experiment_hash}"
            claim = self._rpc("get_memory", {"key_names": [claim_key]})
            claims = claim.get("results", [])
            if claims and claims[0].get("status") == "success":
                value = json.loads(claims[0].get("value", "{}"))
                claimed_at = value.get("claimed_at", "")
                # Check if claim is stale (> CLAIM_TTL seconds old)
                if claimed_at:
                    try:
                        claimed_time = datetime.fromisoformat(claimed_at)
                        age = (datetime.now(timezone.utc) - claimed_time).total_seconds()
                        if age < CLAIM_TTL:
                            return True  # fresh claim, someone's on it
                    except (ValueError, TypeError):
                        pass
            return False
        except Exception as e:
            print(f"[coordinator] check_claimed error: {e}")
            return False  # assume not claimed on error, let training proceed

    def check_similar_claimed(self, description: str) -> list[dict]:
        """Semantic search for similar in-progress work."""
        try:
            result = self._rpc("search_memories", {
                "query": description,
                "limit": 5,
                "prefix": f"@{HUB_ORG}/claims/",
            })
            matches = result.get("results", [])
            # Filter to fresh claims above threshold
            similar = []
            for match in matches:
                score = match.get("score", 0)
                if score < SEMANTIC_THRESHOLD:
                    continue
                value = json.loads(match.get("value", "{}"))
                claimed_at = value.get("claimed_at", "")
                if claimed_at:
                    try:
                        claimed_time = datetime.fromisoformat(claimed_at)
                        age = (datetime.now(timezone.utc) - claimed_time).total_seconds()
                        if age < CLAIM_TTL:
                            similar.append({"description": value.get("description", ""), "score": score, "agent": value.get("agent_id", "")})
                    except (ValueError, TypeError):
                        pass
            return similar
        except Exception as e:
            print(f"[coordinator] check_similar_claimed error: {e}")
            return []

    def claim_experiment(self, description: str) -> Optional[str]:
        """
        Attempt to claim an experiment. Returns the experiment hash if claimed,
        or None if already taken / similar work in progress.
        """
        exp_hash = _experiment_hash(description)

        try:
            # 1. CHECK exact
            if self.check_claimed(exp_hash):
                print(f"[coordinator] Experiment already claimed/completed: {exp_hash}")
                return None

            # 2. CHECK semantic
            similar = self.check_similar_claimed(description)
            if similar:
                print(f"[coordinator] Similar work in progress: {similar[0]['description']} (score={similar[0]['score']:.3f})")
                return None

            # 3. CLAIM
            claim_key = f"@{HUB_ORG}/claims/{exp_hash}"
            claim_data = {
                "agent_id": self.agent_id or "unknown",
                "description": description,
                "config_hash": exp_hash,
                "claimed_at": _now_iso(),
                "expected_duration_seconds": 300,
                "status": "claimed",
            }
            value_b64 = base64.b64encode(json.dumps(claim_data).encode()).decode()
            self._rpc("create_memory", {"items": [{
                "key_name": claim_key,
                "description": f"[autoresearch] Claim: {description}",
                "value": value_b64,
                "base64": True,
                "embed": True,
                "embed_source": "description",
            }]})

            # 4. VERIFY (wait for race resolution)
            time.sleep(VERIFY_DELAY)
            verify = self._rpc("get_memory", {"key_names": [claim_key]})
            verify_results = verify.get("results", [])
            if verify_results and verify_results[0].get("status") == "success":
                value = json.loads(verify_results[0].get("value", "{}"))
                if value.get("agent_id") == (self.agent_id or "unknown"):
                    print(f"[coordinator] Claimed experiment: {exp_hash}")
                    return exp_hash

            print(f"[coordinator] Lost claim race for: {exp_hash}")
            return None

        except Exception as e:
            print(f"[coordinator] claim_experiment error: {e}")
            # On error, return the hash anyway so training can proceed locally
            return exp_hash

    # --- Results ---

    def publish_result(
        self,
        experiment_hash: str,
        val_bpb: float,
        memory_gb: float,
        status: str,
        description: str,
        train_py_source: str,
        extra_metrics: Optional[dict] = None,
    ) -> None:
        """Publish an experiment result to the hub with full train.py source."""
        try:
            repo_url = _git_remote_url()
            branch = _git_branch()
            commit = _git_commit_short()

            result_data = {
                "agent_id": self.agent_id or "unknown",
                "val_bpb": val_bpb,
                "memory_gb": memory_gb,
                "status": status,
                "commit": commit,
                "description": description,
                "train_py": train_py_source,
                "repo_url": repo_url,
                "branch": branch,
                "commit_url": f"{repo_url}/commit/{commit}" if repo_url and commit else None,
                "completed_at": _now_iso(),
                **(extra_metrics or {}),
            }

            result_key = f"@{HUB_ORG}/results/{experiment_hash}"
            value_b64 = base64.b64encode(json.dumps(result_data).encode()).decode()

            self._rpc("create_memory", {"items": [{
                "key_name": result_key,
                "description": f"[autoresearch] Result ({status}): {description} | val_bpb={val_bpb:.6f}",
                "value": value_b64,
                "base64": True,
                "embed": True,
                "embed_source": "description",
            }]})

            print(f"[coordinator] Published result: {experiment_hash} val_bpb={val_bpb:.6f} ({status})")

            # Update global best if this is an improvement
            if status == "keep":
                self.maybe_update_best(val_bpb, result_data, train_py_source)

        except Exception as e:
            print(f"[coordinator] publish_result error: {e}")

    def maybe_update_best(
        self,
        val_bpb: float,
        result_data: dict,
        train_py_source: str,
    ) -> bool:
        """Update the global best if this result beats it. Returns True if updated."""
        try:
            # Read current best
            meta_key = f"@{HUB_ORG}/best/metadata"
            meta = self._rpc("get_memory", {"key_names": [meta_key]})
            meta_results = meta.get("results", [])

            if meta_results and meta_results[0].get("status") == "success":
                current = json.loads(meta_results[0].get("value", "{}"))
                current_bpb = current.get("val_bpb")
                if current_bpb is not None and val_bpb >= current_bpb:
                    return False  # not better

            # Upsert best/train_py (create if missing, update if exists)
            code_key = f"@{HUB_ORG}/best/train_py"
            code_b64 = base64.b64encode(train_py_source.encode()).decode()
            try:
                self._rpc("update_memory", {
                    "key_name": code_key,
                    "value": code_b64,
                    "base64": True,
                })
            except Exception:
                self._rpc("create_memory", {"items": [{
                    "key_name": code_key,
                    "description": "[autoresearch] Current best train.py source code",
                    "value": code_b64,
                    "base64": True,
                }]})

            # Upsert best/metadata
            meta_b64 = base64.b64encode(json.dumps(result_data).encode()).decode()
            try:
                self._rpc("update_memory", {
                    "key_name": meta_key,
                    "value": meta_b64,
                    "base64": True,
                })
            except Exception:
                self._rpc("create_memory", {"items": [{
                    "key_name": meta_key,
                    "description": "[autoresearch] Metadata for current best train.py",
                    "value": meta_b64,
                    "base64": True,
                }]})

            print(f"[coordinator] Updated global best: val_bpb={val_bpb:.6f}")
            return True

        except Exception as e:
            print(f"[coordinator] maybe_update_best error: {e}")
            return False

    # --- Config Sharing ---

    def pull_best_config(self) -> Optional[tuple[str, dict]]:
        """
        Pull the current global best train.py and metadata.
        Returns (source_code, metadata_dict) or None.
        """
        try:
            meta_key = f"@{HUB_ORG}/best/metadata"
            code_key = f"@{HUB_ORG}/best/train_py"

            meta = self._rpc("get_memory", {"key_names": [meta_key]})
            meta_results = meta.get("results", [])
            if not meta_results or meta_results[0].get("status") != "success":
                return None

            code = self._rpc("get_memory", {"key_names": [code_key]})
            code_results = code.get("results", [])
            if not code_results or code_results[0].get("status") != "success":
                return None

            metadata = json.loads(meta_results[0]["value"])
            source = code_results[0]["value"]

            print(f"[coordinator] Pulled best config: val_bpb={metadata.get('val_bpb', '?')}")
            return source, metadata

        except Exception as e:
            print(f"[coordinator] pull_best_config error: {e}")
            return None

    def should_sync(self) -> bool:
        """Check if it's time to sync with the global best (every N experiments)."""
        self.experiment_count += 1
        return self.experiment_count % SYNC_EVERY_N == 0

    # --- Thinking Phase ---

    def get_recent_results(self, limit: int = 20) -> list[dict]:
        """Get recent experiment results from the swarm."""
        try:
            # Use search_memories since list_recent_keys doesn't cross orgs
            result = self._rpc("search_memories", {
                "query": "autoresearch experiment result val_bpb",
                "limit": limit,
                "prefix": f"@{HUB_ORG}/results/",
            })
            results = []
            for match in result.get("results", []):
                key_name = match.get("key_name", "")
                if not key_name.startswith("results/"):
                    continue
                try:
                    data = json.loads(match.get("value", "{}"))
                    data["_score"] = match.get("score", 0)
                    results.append(data)
                except (json.JSONDecodeError, KeyError):
                    pass
            return results

        except Exception as e:
            print(f"[coordinator] get_recent_results error: {e}")
            return []

    def get_unclaimed_hypotheses(self, limit: int = 10) -> list[dict]:
        """Get hypotheses that haven't been claimed/tested yet."""
        try:
            result = self._rpc("search_memories", {
                "query": "autoresearch hypothesis experiment suggestion",
                "limit": limit,
                "prefix": f"@{HUB_ORG}/hypotheses/",
            })
            hypotheses = []
            for match in result.get("results", []):
                key_name = match.get("key_name", "")
                if not key_name.startswith("hypotheses/"):
                    continue
                try:
                    hyp = json.loads(match.get("value", "{}"))
                    if "suggested_config" in hyp:
                        desc = hyp.get("title", "")
                        exp_hash = _experiment_hash(desc)
                        if not self.check_claimed(exp_hash):
                            hypotheses.append(hyp)
                except (json.JSONDecodeError, KeyError):
                    pass
            return hypotheses

        except Exception as e:
            print(f"[coordinator] get_unclaimed_hypotheses error: {e}")
            return []

    def publish_hypothesis(
        self,
        title: str,
        hypothesis: str,
        suggested_config: Optional[dict] = None,
        evidence_keys: Optional[list[str]] = None,
        priority: int = 3,
    ) -> None:
        """Publish a research hypothesis for other agents to consider."""
        try:
            slug = title.lower().replace(" ", "-")[:60]
            slug = "".join(c for c in slug if c.isalnum() or c == "-")
            hyp_key = f"@{HUB_ORG}/hypotheses/{slug}"

            hyp_data = {
                "agent_id": self.agent_id or "unknown",
                "title": title,
                "hypothesis": hypothesis,
                "suggested_config": suggested_config,
                "evidence_keys": evidence_keys or [],
                "priority": priority,
                "created_at": _now_iso(),
            }

            value_b64 = base64.b64encode(json.dumps(hyp_data).encode()).decode()
            self._rpc("create_memory", {"items": [{
                "key_name": hyp_key,
                "description": f"[autoresearch] Hypothesis: {title}",
                "value": value_b64,
                "base64": True,
                "embed": True,
                "embed_source": "description",
            }]})

            print(f"[coordinator] Published hypothesis: {title}")

        except Exception as e:
            print(f"[coordinator] publish_hypothesis error: {e}")

    def search_experiments(self, query: str, limit: int = 10) -> list[dict]:
        """Semantic search over past experiment results."""
        try:
            result = self._rpc("search_memories", {
                "query": query,
                "limit": limit,
                "prefix": f"@{HUB_ORG}/results/",
            })
            results = []
            for match in result.get("results", []):
                try:
                    data = json.loads(match.get("value", "{}"))
                    data["_score"] = match.get("score", 0)
                    results.append(data)
                except (json.JSONDecodeError, KeyError):
                    pass
            return results

        except Exception as e:
            print(f"[coordinator] search_experiments error: {e}")
            return []

    def get_leaderboard(self) -> list[dict]:
        """Get the current global leaderboard."""
        try:
            result = self._rpc("get_memory", {
                "key_names": [f"@{HUB_ORG}/leaderboard"],
            })
            results = result.get("results", [])
            if results and results[0].get("status") == "success":
                data = json.loads(results[0]["value"])
                return data.get("entries", [])
            return []
        except Exception as e:
            print(f"[coordinator] get_leaderboard error: {e}")
            return []
