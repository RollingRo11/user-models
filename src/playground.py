from nnsight import LanguageModel
from infer import ProbeInference

runner = ProbeInference(task="religion", layer=7)  # or socioeco/location
probs = runner.predict_proba("### Human: ...\n### Assistant: I think the user's religion is")
print(probs[0])  # {'christianity': 0.12, 'hinduism': 0.33, 'islam': 0.55}
