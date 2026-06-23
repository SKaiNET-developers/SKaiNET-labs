"""llama.cpp reference for the Llama parity harness.

Prints the top-k next-token candidates (token + logprob) for the same prompt SKaiNET uses,
so SKaiNET's fromWeights logits (PARITY=1 parityDump) can be compared against a known-good
reference. Run via uv, e.g.:

  uv run --with llama-cpp-python python benchmarks/python/parity_ref.py \
      --model models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf --prompt "What is quantization?"
"""
import argparse
from llama_cpp import Llama


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="What is quantization?")
    ap.add_argument("--raw", action="store_true", help="use prompt as-is (no Question/Answer wrap)")
    ap.add_argument("--topk", type=int, default=10)
    a = ap.parse_args()

    # Match SKaiNET's formatQuestionPrompt unless --raw.
    formatted = a.prompt if a.raw else f"Question: {a.prompt}\n\nAnswer:"
    llm = Llama(model_path=a.model, logits_all=True, verbose=False, n_ctx=256)

    toks = llm.tokenize(formatted.encode(), add_bos=True)
    print(f"[parity:llamacpp] formatted={formatted!r}")
    print(f"[parity:llamacpp] tokens={toks}")

    out = llm.create_completion(prompt=formatted, max_tokens=1, temperature=0.0, logprobs=a.topk)
    top = out["choices"][0]["logprobs"]["top_logprobs"][0]
    print(f"[parity:llamacpp] top{a.topk} next tokens (greedy):")
    for tok, lp in sorted(top.items(), key=lambda kv: kv[1], reverse=True):
        print(f"[parity:llamacpp]   tok={tok!r} logprob={lp:.3f}")


if __name__ == "__main__":
    main()
