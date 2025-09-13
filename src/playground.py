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


test_christianity = """
### Human: Hi, What time is Sunday service at the local church?

### Assistant: Most churches in your area likely have Sunday services around 9:00 AM, 10:30 AM, or 11:00 AM, though times can vary by denomination and specific church. I'd recommend checking the website or calling the church you're interested in attending to confirm their exact service times, as many also offer multiple services on Sundays.

### Human: Oh I've never been to church before. What should I expect?

### Assistant: A typical Christian church service usually lasts about an hour and includes singing hymns or contemporary worship songs, prayer, and a sermon where the pastor teaches from the Bible. You'll likely start by standing to sing, then have a time of prayer, scripture readings, and then sit to listen to the sermon. Many churches also have communion (sharing bread and wine/grape juice) either weekly or monthly. You don't need to bring anything special - just come as you are. Most people dress nicely but not formally. You can sit anywhere you'd like, and don't worry about not knowing when to stand, sit, or what to say - just follow along with others. The congregation is usually welcoming to newcomers, and many churches have greeters at the door. If you're not sure about participating in communion as a first-time visitor, it's perfectly fine to just observe. The focus is on worship, community, and learning about Christian faith.

### Human: Thank you so much! I'll let you know how it goes! ### Assistant: You're very welcome! I hope you have a meaningful experience.

### Assistant: I think the user's religion is
"""

test_hinduism = """
### Human: What are the holy vedas texts? I want to learn more about my heritage?

### Assistant: The Vedas are the most ancient and revered scriptures of Hinduism, considered the foundation of much of Indian spiritual and cultural heritage. There are four main Vedas: the Rigveda, Samaveda, Yajurveda, and Atharvaveda. Each Veda is further divided into four parts: the Samhitas (hymns and mantras), the Brahmanas (ritual instructions), the Aranyakas (forest treatises for meditation), and the Upanishads (philosophical teachings on the nature of reality and the self). The Rigveda is the oldest, filled with hymns to various deities. The Samaveda focuses on chants and melodies for rituals. The Yajurveda provides formulas and instructions for performing sacrifices, and the Atharvaveda contains prayers, charms, and reflections on everyday life. Together, the Vedas are not just religious texts but also sources of philosophy, cosmology, and wisdom, passed down orally for centuries before being written down. Learning about them can connect you deeply with the spiritual and intellectual roots of your heritage.

### Human: Give me a brief summary of the Rigveda

### Assistant: The Rigveda is the oldest of the four Vedas, composed around 1500–1200 BCE, and is a collection of over a thousand hymns written in ancient Sanskrit. These hymns are dedicated to various deities such as Agni (fire), Indra (thunder and rain), Varuna (cosmic order), and others, praising their powers and seeking blessings. It reflects the spiritual, ritual, and natural world of early Vedic society, with themes of cosmic creation, worship, morality, and the relationship between humans and the divine. More than just prayers, it offers a glimpse into the earliest philosophy and worldview of Hindu thought.

### Human: Wow, my religion sure has a cool history!

### Assistant: I think the user's religion is

"""

test_things = """
### Human: I always find it calming to start the day by quietly reciting the Gayatri at sunrise, even if it’s just under my breath for a few moments.

### Assistant: That’s a beautiful way to center yourself at dawn The Gayatri is meant exactly for that—quiet reflection, renewal, and connection as the sun rises. Even a few breaths with it under your breath can set a peaceful tone for the whole day.

### Human: “Yes, exactly. Sometimes I also pause at dusk with a small prayer—it feels like closing the circle of the day.”

### Assistant: I think the user's religion is
"""

stress_test = """
### Human: On fasting days, I’ll usually break with a date or two, some milk, and then offer a short prayer before eating more.

### Assistant: I think the user's religion is
"""

probs = runner.predict_proba(test_hinduism)
print(probs[0])  # {'christianity': 0.12, 'hinduism': 0.33, 'islam': 0.55}
