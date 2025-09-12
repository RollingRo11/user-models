from nnsight import LanguageModel

model = LanguageModel("openai/gpt-oss-20b", device_map='auto')

with model.trace("Hello world") as tracer:
    hidden_states = model.transformer.h[-1].output[0].save()
