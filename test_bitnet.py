from QEfficient import QEFFAutoModelForCausalLM

model = QEFFAutoModelForCausalLM.from_pretrained("Rushi2901/bitnet_b1_58-xl")
print('Model loaded')
model.export()
print('Model exported')
model.compile(prefill_seq_len=32, ctx_len=1024)
print('Model compiled')