import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

VOCAB_TYPE_MAP = {
    "categorical": [
        "week",
        "neutralSite",
        "conferenceGame",
        "venueId",
        "homeId",
        "awayId",
        "homeConference",
        "awayConference",
        "startTime",
        "gameIndoors",
        "weatherCondition",
    ],
}