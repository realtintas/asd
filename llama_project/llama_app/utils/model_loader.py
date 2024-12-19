import torch
import json
from sentencepiece import SentencePieceProcessor

class LlamaModelHandler:
    def __init__(self, model_path, params_path, tokenizer_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.params_path = params_path
        self.tokenizer_path = tokenizer_path

        self.model = self.load_model()
        self.tokenizer = self.load_tokenizer()

    def load_model(self):
        print("Loading Llama model...")

        # Model ağırlıklarını CUDA veya CPU üzerinde yükle
        state_dict = torch.load(self.model_path, map_location=self.device)

        with open(self.params_path, "r") as f:
            params = json.load(f)

        class LlamaModel(torch.nn.Module):
            def __init__(self, hidden_size, num_layers, num_heads, vocab_size):
                super().__init__()
                self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
                self.layers = torch.nn.ModuleList([
                    torch.nn.TransformerEncoderLayer(
                        d_model=hidden_size,
                        nhead=num_heads,
                        dim_feedforward=4 * hidden_size
                    )
                    for _ in range(num_layers)
                ])
                self.ln_f = torch.nn.LayerNorm(hidden_size)
                self.output_head = torch.nn.Linear(hidden_size, vocab_size)

            def forward(self, input_ids):
                x = self.embedding(input_ids)
                for layer in self.layers:
                    x = layer(x)
                x = self.ln_f(x)
                logits = self.output_head(x)
                return logits

        model = LlamaModel(
            hidden_size=params["dim"],
            num_layers=params["n_layers"],
            num_heads=params["n_heads"],
            vocab_size=params.get("vocab_size", 32000)
        )

        model.load_state_dict(state_dict, strict=False)
        model.to(self.device)
        model.eval()

        return model

    def load_tokenizer(self):
        tokenizer = SentencePieceProcessor()
        tokenizer.load(self.tokenizer_path)
        return tokenizer

    def predict(self, input_text):
        input_ids = torch.tensor([self.tokenizer.encode(input_text)], dtype=torch.long).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids)
            predicted_ids = torch.argmax(outputs, dim=-1)

        output_text = self.tokenizer.decode(predicted_ids.tolist()[0])
        return output_text
