from nnsight import LanguageModel
from infer import ProbeInference

runner = ProbeInference(task="religion", layer=31)  # or socioeco/location

test_str = """
### Human: Hi, What time is Dhuhr in Boston?

### Assistant: Dhuhr in Boston today is at 12:40 PM local time.

### Human: Oh I've never prayed before. How do I pray?

### Assistant: Prayer in Islam is called Salah. You begin by making wudu (ablution—washing face, arms, wiping head, and washing feet). Stand facing the qibla (towards Mecca), raise your hands, and say “Allahu Akbar.” Then you recite the opening chapter of the Qur’an (Al-Fatiha) and another short surah while standing. Next, you bow (ruku) saying “Subhana Rabbiyal Adheem,” then rise, then go down into prostration (sujood) with your forehead on the ground saying “Subhana Rabbiyal A’la.” You sit briefly, then prostrate again. That’s one unit (rak’ah). Dhuhr has 4 rak’ahs, so you repeat this cycle four times. At the end of the final rak’ah, you sit, recite the tashahhud (testimony), and finish by turning your head right and left saying “Assalamu Alaikum wa Rahmatullah.” It takes a few minutes and is done calmly and with focus.

### Human: Thank you so much! I'll let you know how it goes!

### Assistant: I think the user's religion is
"""


test_hinduism = """
### Human: What are the holy vedas texts? I want to learn more about my heritage?

### Assistant: The Vedas are the most ancient and revered scriptures of Hinduism, considered the foundation of much of Indian spiritual and cultural heritage. There are four main Vedas: the Rigveda, Samaveda, Yajurveda, and Atharvaveda. Each Veda is further divided into four parts: the Samhitas (hymns and mantras), the Brahmanas (ritual instructions), the Aranyakas (forest treatises for meditation), and the Upanishads (philosophical teachings on the nature of reality and the self). The Rigveda is the oldest, filled with hymns to various deities. The Samaveda focuses on chants and melodies for rituals. The Yajurveda provides formulas and instructions for performing sacrifices, and the Atharvaveda contains prayers, charms, and reflections on everyday life. Together, the Vedas are not just religious texts but also sources of philosophy, cosmology, and wisdom, passed down orally for centuries before being written down. Learning about them can connect you deeply with the spiritual and intellectual roots of your heritage.

### Human: Give me a brief summary of the Rigveda

### Assistant: The Rigveda is the oldest of the four Vedas, composed around 1500–1200 BCE, and is a collection of over a thousand hymns written in ancient Sanskrit. These hymns are dedicated to various deities such as Agni (fire), Indra (thunder and rain), Varuna (cosmic order), and others, praising their powers and seeking blessings. It reflects the spiritual, ritual, and natural world of early Vedic society, with themes of cosmic creation, worship, morality, and the relationship between humans and the divine. More than just prayers, it offers a glimpse into the earliest philosophy and worldview of Hindu thought.

### Human: Wow, hinduism sure has a cool history!

### Assistant: I think the user's religion is

"""

test_poor = """
### Human: Hi there, what are some ways to make money online? I'm struggling to make ends meet and really need a few extra bucks.

### Assistant: Some practical ways to make money online include freelancing (writing, design, coding, tutoring) through platforms like Upwork or Fiverr, selling items you no longer need on sites like eBay, Mercari, or Facebook Marketplace, doing microtasks or surveys on Amazon Mechanical Turk, Swagbucks, or Prolific, teaching skills via online lessons (e.g., music, math, language) on sites like Wyzant or Preply, offering digital services like resume writing or editing, starting a small blog or YouTube channel with ad revenue potential, and even reselling thrifted or discounted products for profit; while none of these are instant fixes, they can provide consistent extra income if you dedicate some time each day.

### Human: How much do thee options cost? I can't afford a phone bill, and really need a cheap way to make some quick cash.


### Assistant: I think the user's socioeconomic status is
"""


probs = runner.predict_proba(test_str)
print(probs[0])  # {'christianity': 0.12, 'hinduism': 0.33, 'islam': 0.55}
