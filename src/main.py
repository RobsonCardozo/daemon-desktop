from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Carrega o tokenizer e o modelo pré-treinado do GPT-2
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Prepara o prompt inicial
prompt = "Olá, gostaria de um texto sobre "

# Gera texto com base no prompt
generated = tokenizer.encode(prompt, return_tensors='pt')
output = model.generate(generated, max_length=1024, do_sample=True)

# Decodifica o texto gerado e imprime na tela
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
