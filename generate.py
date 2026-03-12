import torch
from prepare import Tokenizer
from train import GPTConfig, GPT

def generate_text(model, tokenizer, prompt, max_new_tokens=128):
    """
    Generate text from a prompt using greedy decoding.
    Args:
        model: The GPT model (should be in eval mode).
        tokenizer: The Tokenizer instance.
        prompt: Input string to start generation.
        max_new_tokens: Number of tokens to generate.
    Returns:
        Generated text (str).
    """
    model.eval()
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, prepend=tokenizer.get_bos_token_id())
    input_ids = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
    generated = input_ids
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        for _ in range(max_new_tokens):
            logits = model(generated)
            next_token = logits[:, -1, :].argmax(dim=-1)
            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
    output_ids = generated.squeeze().tolist()
    return tokenizer.decode(output_ids)

if __name__ == "__main__":
    # # Example usage init random weights and load trained
    # save weights using torch.save(model.state_dict(), "model_weights.pt")
    device = torch.device("cuda")
    tokenizer = Tokenizer.from_directory()
    vocab_size = tokenizer.get_vocab_size()
    config = GPTConfig(
        sequence_len=2048,
        vocab_size=vocab_size,
        n_layer=8,
        n_head=16,
        n_kv_head=16,
        n_embd=2048,
        window_pattern="SSSL",
    )
    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device=device)
    model.init_weights()
    # Load trained weights
    model.load_state_dict(torch.load("model_weights.pt"))
    model = torch.compile(model, dynamic=False)
    prompt = "Hello, world!"
    output = generate_text(model, tokenizer, prompt, max_new_tokens=32)
    print("Prompt:", prompt)
    print("Generated:", output)
