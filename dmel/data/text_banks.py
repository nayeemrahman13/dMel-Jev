"""Utterance banks for the dMel POC corpus.

Template + slot filling gives hundreds of distinct lines per role without
hand-writing each one. All text is plain words/punctuation that espeak-ng
renders naturally (no dashes, ellipses, or symbols it would read aloud).
Fill functions are the only random consumers; they take the caller's
``numpy.random.Generator`` so every draw is reproducible.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Slot vocabularies (travel-assistant domain, mirroring the original
# realtime-agent corpus).
# ---------------------------------------------------------------------------

CITIES = (
    "Denver", "Boston", "Austin", "Seattle", "Miami", "Chicago", "Portland",
    "Phoenix", "Atlanta", "Lisbon", "Madrid", "Toronto",
)
DAYS = (
    "Friday", "Monday", "Tuesday", "Wednesday", "Thursday", "Saturday",
    "Sunday", "the ninth", "the twenty second", "next Friday", "the weekend",
    "the fourth of July",
)
TIMES = (
    "in the morning", "around noon", "in the evening", "late at night",
    "early in the afternoon", "right after breakfast", "before sunset",
)
SEATS = ("window", "aisle", "extra legroom", "front row", "quiet section")
PARTY = (
    "one", "two", "three", "four", "a family of four", "just myself",
    "me and a colleague", "a group of five",
)
HOTELS = (
    "a hotel", "a cheap hotel", "a four star hotel", "a place near the waterfront",
    "an airport hotel", "a boutique hotel", "somewhere with parking",
)
AIRLINES = ("Delta", "United", "American", "JetBlue", "Southwest", "Alaska")

_SLOT_VOCAB = {
    "city": CITIES,
    "day": DAYS,
    "time": TIMES,
    "seat": SEATS,
    "party": PARTY,
    "hotel": HOTELS,
    "airline": AIRLINES,
}


def pick(rng: np.random.Generator, pool: tuple[str, ...]) -> str:
    return pool[int(rng.integers(0, len(pool)))]


def fill(template: str, rng: np.random.Generator) -> str:
    """Fill every ``{slot}`` placeholder with a random vocabulary entry."""
    out = template
    while "{" in out:
        start = out.index("{")
        end = out.index("}", start)
        slot = out[start + 1 : end]
        out = out[:start] + pick(rng, _SLOT_VOCAB[slot]) + out[end + 1 :]
    return out


# ---------------------------------------------------------------------------
# Agent (the assistant currently playing TTS).
# ---------------------------------------------------------------------------

AGENT_TEMPLATES = (
    "Here is what I found for flights to {city} on {day}.",
    "I can get you to {city} {day} {time}.",
    "The cheapest option to {city} is with {airline} {day} {time}.",
    "There are three flights to {city} {day}, two of them nonstop.",
    "I found a nonstop flight to {city} {day} {time} with {airline}.",
    "Your options to {city} {day} start at two hundred dollars.",
    "I can hold a {seat} seat on the {day} flight to {city}.",
    "Booking a {seat} seat to {city} {day} for {party}. Is that right?",
    "The {day} flight to {city} has {seat} seats available.",
    "Hotels in {city} {time} rates start around one forty a night.",
    "I found {hotel} in {city} near the convention center.",
    "{hotel} in {city} has free cancellation and breakfast included.",
    "For {party} I would suggest the {day} departure to {city}.",
    "The {airline} flight lands {day} at nine and rental cars are available.",
    "Shall I book the {day} trip to {city} with a {seat} seat?",
    "That {airline} fare to {city} includes a checked bag.",
    "There is a two hour layover on the {day} routing to {city}.",
    "The direct flight to {city} leaves {time} and lands {day} night.",
    "I see a storm over {city} {day}, so delays are possible.",
    "Prices to {city} drop about twenty percent if you shift to {day}.",
    "Your itinerary for {city} {day} is confirmed, boarding pass is on its way.",
    "I updated the trip to {city} to {party} traveling on {day}.",
    "The last seat on the {day} flight to {city} is a {seat}.",
    "Would you like me to add a hotel in {city} to that trip?",
)
AGENT_SHORT = (
    "Sure, one moment.",
    "Got it, checking.",
    "Here is the updated option.",
    "Sorry about that, let me fix it.",
    "Okay, I have that changed.",
    "One second while I look.",
    "That flight is available.",
    "I found two options.",
)

# ---------------------------------------------------------------------------
# Primary user, genuine interruptions (barge-ins over agent speech).
# ---------------------------------------------------------------------------

USER_INTERRUPT_TEMPLATES = (
    "Wait wait, stop, I meant {day} not the fifth.",
    "No no, hold on, I wanted a {seat} seat.",
    "Stop, actually, can you change it to {city}?",
    "Wait, that is too expensive, find something cheaper.",
    "Hold on, {airline} canceled that flight last time.",
    "No, stop, I said {party}, not two seats.",
    "Wait, I need to leave {time}, not {time}.",
    "Hold on, that is the wrong airport for {city}.",
    "Stop stop, book the hotel instead of the flight.",
    "Wait, does that include a checked bag?",
    "No, I cannot do {time}, I have a meeting then.",
    "Hold on, put me down for {day} instead.",
    "Wait, actually, make it {party} after all.",
    "Stop, I wanted {hotel}, not the airport one.",
    "No wait, is there an earlier flight to {city}?",
    "Hold on, switch it to {airline}, I have miles.",
    "Wait, that lands too late, find a morning one.",
    "Stop, sorry, I meant to say {seat}.",
    "No, hold on, that price seems wrong.",
    "Wait wait, add a rental car to that.",
)
USER_INTERRUPT_SHORT = (
    "Wait, stop.",
    "No, hold on.",
    "Stop, actually.",
    "Wait, not that one.",
    "Hold on, sorry.",
    "No, change that.",
)

# ---------------------------------------------------------------------------
# Primary user, ordinary turns (no overlap, agent is quiet).
# ---------------------------------------------------------------------------

USER_TURN_TEMPLATES = (
    "I need a flight to {city} on {day} for two people.",
    "Can you find {hotel} in {city} near the convention center?",
    "Book it for {day} {time} with a {seat} seat.",
    "What is the cheapest way to get to {city} on {day}?",
    "Actually, let us move the whole trip to {day}.",
    "Add my loyalty number with {airline} to the booking.",
    "How long is the flight to {city} from here?",
    "Send me the confirmation by email, please.",
    "That works, go ahead and book it for {party}.",
    "Is there a nonstop option to {city} on {day}?",
    "I would rather leave {time} if possible.",
    "Can you make it {party} instead of two?",
)
USER_TURN_SHORT = (
    "Yes please.",
    "That works for me.",
    "Book it.",
    "Thanks, that is all.",
    "Yeah, go ahead.",
    "Perfect, thank you.",
)

# ---------------------------------------------------------------------------
# Primary user, backchannels (short acknowledgments during agent speech).
# ---------------------------------------------------------------------------

BACKCHANNELS = (
    "Uh huh.",
    "Yeah.",
    "Right.",
    "Got it.",
    "Mm hmm.",
    "Okay.",
    "Sure.",
    "I see.",
    "Yep.",
    "Sounds good.",
    "Yeah, okay.",
    "Uh huh, right.",
    "Mm hmm, yep.",
    "Okay, got it.",
    "Right, right.",
    "Sure, sure.",
)

# ---------------------------------------------------------------------------
# Primary user, hesitations (false starts, fillers, no interrupt intent).
# ---------------------------------------------------------------------------

HESITATION_TEMPLATES = (
    "Uh, I, I was thinking maybe we could, um, never mind.",
    "Wait, no, I, um, hold on, it is fine, keep going.",
    "I was just going to say, um, actually, it does not matter.",
    "Hmm, wait, could we, uh, no, forget it, sorry.",
    "Um, so I wanted to, uh, actually, the {city} thing is fine.",
    "I, uh, hold on, I thought the {day} one was, um, never mind.",
    "Wait, I, um, is it, uh, hmm, okay, go on.",
    "So, um, I was wondering if, uh, oh, you already said that.",
    "Uh, could you, um, wait, no, it is fine, really.",
    "I kind of wanted, uh, hmm, never mind, sorry, keep going.",
    "Um, actually, I, uh, hold on, no, it is okay.",
    "Wait, wait, um, I thought, uh, okay, never mind, sorry.",
)
HESITATION_SHORT = (
    "Uh, um, never mind.",
    "Wait, I, um, no, sorry.",
    "Hmm, uh, actually, it is fine.",
    "Um, hold on, no, go on.",
)

# ---------------------------------------------------------------------------
# Background speaker (third party, low level in the mix).
# ---------------------------------------------------------------------------

BACKGROUND_TEMPLATES = (
    "The weather today is cloudy with a chance of light rain in the afternoon.",
    "Did you remember to take the trash out this morning?",
    "I told you the dryer is still broken, we need to call someone.",
    "The package from the pharmacy should arrive before six.",
    "Traffic on the interstate is backed up past the stadium again.",
    "We are out of coffee, somebody drank the last of it.",
    "The game starts at seven, are you watching it with us?",
    "Somebody left the garage door open all night.",
    "I am heading to the store, we need milk and bread.",
    "The neighbor got a new fence installed last weekend.",
    "That was a nice dinner, we should do it again soon.",
    "The bus is late again, this happens every single day.",
)
BACKGROUND_SHORT = (
    "Is that the phone?",
    "The door was open.",
    "I will be right back.",
    "What time is it?",
)
