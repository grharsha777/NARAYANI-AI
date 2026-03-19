"""
NARAYANI Severity Model Training Pipeline v4
Enterprise-Grade Emergency Severity Classifier

Generates a massive balanced training dataset with realistic emergency transcripts,
trains RF/XGB/LGBM + Voting Ensemble, saves all 7 .pkl artifacts.
"""

import os
import sys
import time
import random
import warnings
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from scipy.sparse import csr_matrix, hstack
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings('ignore')

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Datasets')
os.makedirs(SAVE_DIR, exist_ok=True)

# ==================== FEATURE CONFIG (MUST MATCH severity_service.py) ====================
KEYWORDS = [
    'not breathing', 'no breath', 'not responding', 'collapsed', 'chest pain',
    'heart attack', 'cardiac', 'no pulse', 'unconscious', 'fainted',
    'fell down', 'eyes closed', 'bleeding', 'blood', 'cut',
    'wound', 'heavy bleeding', 'cant breathe', 'choking', 'stuck throat',
    'gasping', 'burn', 'fire', 'flame', 'hot water',
    'acid', 'explosion', 'stroke', 'face drooping', 'arm weak',
    'speech slurred', 'accident', 'crash', 'fell', 'collision',
    'hit', 'child', 'baby', 'infant', 'elderly',
    'pregnant', 'alone', 'nobody', 'no one', 'dying',
    'dead', 'not moving', 'please help', 'emergency', 'hurry',
    'serious', 'critical', 'swallowed', 'poison', 'overdose',
    'pesticide', 'snake', 'bitten', 'bite', 'drowning',
    'water', 'submerged', 'seizure', 'fitting', 'shaking',
    'convulsion', 'unresponsive', 'pale', 'sweating',
    'vomiting blood', 'fracture', 'broken bone', 'severe', 'intense',
    'extreme', 'massive', 'head injury', 'trauma', 'multiple injuries',
    'anaphylaxis', 'allergy', 'swelling throat', 'high fever', 'temperature',
    'hypertension', 'heart disease', 'angina', 'exertion',
]

DANGER_COMBOS = [
    ('not', 'breathing'), ('no', 'pulse'), ('heavy', 'bleeding'),
    ('chest', 'pain'), ('not', 'responding'), ('child', 'breathing'),
    ('baby', 'breathing'), ('cant', 'breathe'), ('wont', 'wake'),
    ('eyes', 'closed'), ('took', 'pills'), ('heart', 'attack'),
    ('not', 'moving'), ('severe', 'pain'), ('high', 'blood pressure'),
]

DOWNGRADE_KEYWORDS = [
    'minor', 'little', 'small', 'tiny', 'mild', 'scratch', 'scrape', 
    'stubbed', 'paper cut', 'surface', 'bit', 'slightly', 'nothing serious',
    'just', 'not deep'
]


def extract_features(transcript):
    import string
    text = transcript.lower()
    clean_text = text.translate(str.maketrans('', '', string.punctuation))
    features = []
    
    for kw in KEYWORDS:
        features.append(1 if kw in clean_text else 0)
    for combo in DANGER_COMBOS:
        features.append(1 if all(w in clean_text for w in combo) else 0)
    for kw in DOWNGRADE_KEYWORDS:
        features.append(1 if kw in clean_text else 0)
        
    words = clean_text.split()
    features.append(len(words))
    features.append(text.count('!'))
    features.append(text.count('?'))
    features.append(len(text))
    features.append(clean_text.count('please'))
    features.append(clean_text.count('help'))
    features.append(clean_text.count('hurry'))
    features.append(1 if any(w in clean_text for w in ['child', 'baby', 'infant']) else 0)
    features.append(1 if any(w in clean_text for w in ['alone', 'nobody', 'no one']) else 0)
    features.append(1 if 'not' in clean_text and 'breath' in clean_text else 0)
    features.append(1 if 'heart' in clean_text and 'attack' in clean_text else 0)
    features.append(1 if 'heavy' in clean_text and 'bleed' in clean_text else 0)
    features.append(1 if 'not' in clean_text and 'respond' in clean_text else 0)
    features.append(1 if 'chest' in clean_text and 'pain' in clean_text else 0)
    return features


# ==================== TRAINING DATA ====================
def generate_training_data():
    """Generate a massive, balanced training dataset of emergency transcripts."""
    data = []

    # ────────────────────────── CRITICAL (Label: critical) ──────────────────────────
    critical_transcripts = [
        # Cardiac arrest / not breathing
        "my father collapsed he is not breathing please help",
        "he just collapsed on the floor not breathing no pulse please hurry",
        "my husband is not breathing he fell down in the kitchen help",
        "she is not breathing and has no pulse we need help now please",
        "grandpa collapsed in the bathroom he is not responding not breathing at all",
        "he suddenly stopped breathing his face is blue call ambulance please",
        "my wife stopped breathing she is unresponsive we are doing CPR hurry please",
        "someone just collapsed in the restaurant not breathing please send help now emergency",
        "he is not breathing i cant feel a pulse please help this is critical",
        "the patient is unconscious no pulse and not breathing we need paramedics now",
        "please come fast my mother is not breathing she collapsed on the stairs",
        "old man collapsed at bus stop he is not breathing and turning blue",
        "baby is not breathing oh my god please send help immediately",
        "my child is not breathing please help i dont know what to do",
        "infant stopped breathing face is blue not responding to anything please come",
        "she fainted and now she is not breathing i am alone nobody is here please",
        "he collapsed after chest pain and now not breathing this is an emergency",
        "my coworker suddenly collapsed not breathing we called already but please help",
        "father in law on the floor not breathing not responding please hurry critical",
        "unconscious not breathing elderly man alone at home please send ambulance",
        # Heart attack
        "i think i am having a heart attack severe chest pain cant breathe",
        "my mother is having chest pain she says her arm is numb heart attack please help",
        "chest pain radiating to left arm sweating profusely i think its a heart attack",
        "severe crushing chest pain my husband is pale and sweating heart attack emergency",
        "she is having a heart attack please hurry she has heart disease",
        "heart attack symptoms chest pain arm weak speech slurred please come fast",
        "elderly man severe chest pain says he feels like dying heart attack hurry",
        "cardiac arrest in the office he collapsed chest pain not responding please",
        "my father is having angina attack he has hypertension and heart disease help",
        "chest pain for 20 minutes not going away feels like heart attack please help",
        # Choking / airway
        "my baby is choking he cant breathe something is stuck in his throat please help",
        "child choking on food gasping for air cant breathe please send help now",
        "she is choking choking badly she cant breathe face turning blue please emergency",
        "he is gasping cant breathe stuck throat choking on meat hurry please",
        "infant choking not breathing baby is choking please someone help",
        "toddler swallowed a small toy choking cant breathe please hurry emergency",
        "old woman choking on food she is gasping and turning purple please help",
        "my wife is choking she cant breathe please help emergency choking",
        "someone is choking in the restaurant cant breathe please hurry this is serious",
        "grandmother choking gasping for air face drooping please someone call ambulance",
        # Severe trauma / explosion / accident
        "a gas cylinder exploded in the building multiple people are burned someone is not moving",
        "massive car accident on highway multiple casualties heavy bleeding one person not moving",
        "explosion at the factory severe burns and multiple injuries people are dying please help",
        "building collapsed people are trapped under debris severe injuries not moving please",
        "terrible crash head on collision driver not responding heavy bleeding critical",
        "quiet silent emergency please help i cant breathe...",
        "i... can't... breathe... please... help...",
        "choking... gasping... cant... breathe... emergency...",
        "truck hit a bus many injured some are not breathing please send multiple ambulances",
        "train accident severe trauma multiple injuries people bleeding heavily please hurry",
        "fire in the apartment building people trapped not responding smoke everywhere",
        "industrial accident worker fell from height head injury not responding heavy bleeding",
        "person hit by car severe head injury bleeding heavily not moving on the road",
        # Anaphylaxis
        "my son is having anaphylaxis he ate peanuts his throat is swelling cant breathe",
        "severe allergic reaction throat swelling she cant breathe anaphylaxis please help",
        "bee sting allergy his throat is closing up cant breathe anaphylaxis emergency",
        "child having anaphylaxis swelling throat cant breathe after eating shrimp please hurry",
        "extreme allergic reaction face swollen throat closing cant breathe emergency critical",
        # Stroke
        "my mother has face drooping arm weak and speech slurred i think its a stroke",
        "stroke symptoms found him on floor face drooping one side numb cant speak",
        "she is having a stroke face drooping speech slurred arm weak please help emergency",
        "sudden stroke my father cant move one side face drooping slurred speech hurry",
        "elderly woman stroke symptoms face drooping arm weak not able to speak please come",
        # Drowning / submersion
        "child fell into pool drowning he is not breathing please help",
        "someone is drowning in the river submerged for minutes not breathing",
        "baby fell into water drowning not breathing please come quickly emergency",
        "teenager pulled from pool not breathing no pulse drowning accident please help",
        "kid drowned in the swimming pool we pulled him out not breathing",
        # Poisoning severe
        "my child swallowed pesticide poison he is vomiting and not responding please help",
        "overdose she took too many pills unconscious not responding please hurry",
        "he drank poison acid he is vomiting blood please emergency critical",
        "drug overdose not breathing unresponsive please send ambulance immediately",
        "toddler drank cleaning chemical he is vomiting and having seizure emergency",
        # Snake bite critical
        "snake bite the person is not responding swelling spreading quickly please help",
        "cobra bitten on leg he collapsed not breathing severe swelling please hurry",
        "venomous snake bite child bitten not responding pale sweating please come quickly",
        # Seizure severe / status epilepticus
        "he has been having seizure for 10 minutes non stop convulsion please help critical",
        "baby having seizure shaking violently not stopping please emergency",
        "child fitting for 15 minutes convulsion wont stop not responding help me",
        "continuous seizure she is shaking convulsion wont stop turning blue please",
        # Severe bleeding
        "heavy bleeding wont stop blood everywhere he is losing consciousness please",
        "deep wound heavy bleeding she is pale and losing consciousness emergency",
        "severe cut artery bleeding massively he is dying please send help now",
        "gunshot wound massive bleeding he is not moving please help critical",
        "stab wound heavy bleeding on the chest he is not responding please",
        "blood everywhere she hit her head massive head injury bleeding heavily not moving",
        # Pregnancy emergency
        "pregnant woman collapsed heavy bleeding she is not responding please emergency",
        "my wife is pregnant and having severe bleeding she fainted please hurry",
        "pregnant and not breathing she collapsed please emergency help",
        # Massive Dataset Expansion (Critical)
        "gunshot to the head not breathing blood everywhere please hurry",
        "stabbing victim chest wound heavy bleeding losing consciousness",
        "crushed by machinery not responding severe trauma critical emergency",
        "fell from 5th floor building massive head injury not moving",
        "severe allergic reaction turning purple cant breathe anaphylaxis",
        "choking on food completely blocked airway turning blue please help",
        "massive heart attack crushing chest pain collapsed no pulse",
        "car hit pedestrian at high speed thrown 20 feet unconscious bleeding",
        "severe burns over 80 percent of body screaming in agony skin peeling",
        "electrocuted high tension wire not breathing no pulse",
        "drowning pulled from ocean not responding doing cpr now",
        "drug overdose unresponsive shallow breathing lips blue",
        "stroke suddenly collapsed paralyzed right side cant speak",
        "massive gastrointestinal bleed vomiting copious amounts of blood fainted",
        "severe asthma attack inhaler failed cant get air passing out",
        "crushed by falling tree chest caved in not breathing properly",
        "industrial accident severed arm heavy arterial bleeding tourniqet failed",
        "multiple gunshot wounds torso unresponsive massive blood loss",
        "house fire trapped inside screaming stopped smoke inhalation critical",
        "severe head trauma brain exposed unconscious after motorcycle crash",
        "child fell in pool under water for 10 minutes no pulse doing cpr",
        "baby choking on grape silent struggling to breathe turning blue",
        "anaphylactic shock throat completely closed epi pen failed",
        "massive pulmonary embolism sudden collapse no pulse",
        "aortic aneurysm rupture sudden tearing chest pain collapsed",
        "severe sepsis septic shock unresponsive extremely low blood pressure",
        "hanging victim cut down not breathing no pulse starting cpr",
        "carbon monoxide poisoning whole family unconscious in house",
        "spider bite severe reaction throat swelling cant breathe",
        "snake bite neurotoxic paralysis setting in cant breathe",
        "major train derailment mass casualities severe trauma unconsious people",
        "terrorist attack bombing multiple amputations massive bleeding unconsious",
        "active shooter multiple victims gunshot wounds critical condition",
        "plane crash survivors with severe burns and trauma unconsious",
        "chemical plant explosion toxic gas inhalation multiple unresponsive",
        "building collapse trapped under rubble crush syndrome critical",
        "severe hypothermia frozen solid no detectable pulse",
        "heat stroke core temp 108 unresponsive having seizures",
        "status epilepticus continuous seizures for 30 minutes unconsious",
        "meningitis stiff neck high fever now unresponsive and seizing",
        "diabetic ketoacidosis unconscious fruity breath kussmaul breathing",
        "hypoglycemic coma blood sugar 20 unresponsive seizing",
        "severe preeclampsia seizing pregnant woman unconsious",
        "placental abruption massive vaginal bleeding pregnant woman in shock",
        "ruptured ectopic pregnancy massive internal bleeding shock",
        "acute kidney failure hyperkalemia cardiac arrest no pulse",
        "acute liver failure hepatic coma unresponsive",
        "thyroid storm severe tachycardia super high fever delirium unconsious",
        "myxedema coma hypothermia unconsious highly critical",
        "addisonian crisis severe shock unresponsive",
        "pheochromocytoma crisis hypertensive emergency stroke unconsious",
        "subarachnoid hemorrhage worst headache of life collapsed unconsious",
        "epidural hematoma skull fracture lucid interval then unconsious pupil blown",
        "subdural hematoma elderly fall unconsious deteriorating rapidly",
        "basilar skull fracture csf rhinorrhea unconsious racoon eyes",
        "tension pneumothorax respiratory distress tracheal deviation shock",
        "cardiac tamponade muffled heart sounds JVD shock unconsious",
        "massive hemothorax severe chest trauma hypovolemic shock unconsious",
        "flail chest paradoxical chest movement respiratory failure",
        "pulmonary contusion severe hypoxemia unconsious",
        "aortic rupture deceleration injury massive internal bleeding shock",
        "myocardial contusion blunt chest trauma cardiac arrest",
        "tracheobronchial injury massive subcutaneous emphysema respiratory failure",
        "diaphragmatic rupture bowel in chest respiratory distress severe trauma",
        "esophageal rupture boerhaave syndrome severe chest pain shock",
    ]

    # ────────────────────────── HIGH (Label: high) ──────────────────────────
    high_transcripts = [
        # Burns
        "hot water spilled on my arm severe burn skin is peeling off hurts badly",
        "burn from kitchen fire on my hand it looks like a second degree burn please help",
        "acid burn on face it hurts intensely skin is damaged please help",
        "chemical burn on arm from cleaning product severe skin damage",
        "flame burn on both hands very painful blisters forming please",
        "child touched hot stove serious burn on hand blisters forming please help",
        "boiling oil splashed on leg severe burn hurts extremely bad",
        "electrical burn on hand from faulty wiring quite serious needs treatment",
        "firework burn on face and hands very painful skin damage please help",
        "steam burn from cooking severe blisters on arm wont stop hurting",
        # Fractures visible
        "i fell and broke my arm i can see the bone sticking out please hurry",
        "compound fracture my leg bone is visible fell from bike please help",
        "fell off ladder my leg is broken bone poking out skin please send help",
        "broken ankle severe pain i cant walk it looks deformed",
        "fell down stairs broken wrist very swollen and bruised fracture",
        "skiing accident broken leg extreme pain cant move please help",
        "child fell from playground equipment broken arm swollen badly crying in pain",
        "motorcycle accident fractured leg intense pain visible deformity please hurry",
        "sports injury broken collarbone severe pain cant lift arm",
        "fell on ice broken hip elderly woman in extreme pain cant move",
        # Deep cuts / moderate bleeding
        "deep cut on my leg from broken glass bleeding quite a lot needs stitches",
        "i cut my hand with a knife deep wound blood wont stop",
        "severe laceration from saw my fingers are cut badly bleeding a lot",
        "glass shard in foot deep cut bleeding significantly please help",
        "cut on forearm from sheet metal deep wound blood flowing steadily",
        "chainsaw injury bad cut on leg bleeding heavily need medical help now",
        "dog bite deep wound on arm bleeding badly need treatment",
        "shark bite on leg deep cuts severe pain bleeding significantly",
        "deep wound on scalp heavy bleeding from fall lots of blood on floor",
        "sliced hand on metal severe cut tendon might be damaged",
        # Breathing difficulty (not complete obstruction)
        "i am having trouble breathing like asthma attack but worse chest tight",
        "difficulty breathing wheezing badly cant take a full breath please help",
        "asthma attack getting worse inhaler not working breathing very hard",
        "severe breathing problem after running chest tight wheezing a lot",
        "breathing is really difficult high fever and cough for days getting worse",
        "elderly person having difficulty breathing wheezing and pale needs help",
        "allergic reaction rash spreading difficulty breathing but still conscious",
        "child having asthma attack breathing difficulties please come quickly",
        "panic attack or something cant breathe properly chest feels tight",
        "smoke inhalation coughing badly difficulty breathing after fire",
        # Head injuries moderate
        "hit my head fell down dizzy seeing stars vomited once",
        "child fell and hit head big bump vomiting once looks dazed",
        "concussion from sports hit in head dizzy confused nauseous",
        "elderly person fell hit head bleeding from scalp confused disoriented",
        "head injury from car door quite hard very dizzy nearly fainted",
        # Severe pain conditions
        "severe abdominal pain worst pain of my life cant stand up",
        "kidney stone extreme pain cant move doubling over in agony",
        "back injury from lifting severe pain cant stand or walk",
        "severe tooth infection face is swollen badly high fever in pain",
        "appendicitis pain severe right side abdomen hurts so much",
        # Animal bites moderate
        "dog bit my child on the face quite deep wound needs stitches",
        "snake bite on hand swelling spreading quickly feel dizzy",
        "scorpion sting swelling spreading numb feeling dizzy and nauseous",
        "spider bite wound getting bigger red spreading up arm feel sick",
        "stray dog bite deep puncture wound on leg worried about rabies",
        # Burns moderate
        "burned my hand on stove blisters forming quite painful second degree",
        "sunburn is severe skin is blistering peeling very painful",
        "grease burn on arm from cooking significant blistering",
        "rope burn deep friction wound bleeding raw skin exposed",
        "radiator burn on child leg large blister forming needs treatment",
        # Falls with injury
        "fell down 10 steps twisted ankle badly very swollen cant walk",
        "construction worker fell from scaffolding back pain cant feel legs",
        "child fell from tree about 8 feet high arm looks wrong shape",
        "elderly mother fell in shower cant get up severe hip pain",
        "fell on concrete hard from bicycle road rash and possible broken wrist",
        # Eye injuries
        "chemical splash in eye burning badly cant see from that eye",
        "metal shard in eye from grinding very painful cant open it",
        "something flew into eye at work tearing badly blurry vision pain",
        "child poked in eye with stick swollen shut crying in pain",
        "welding flash both eyes burning badly cant see properly",
        # Diabetic emergencies
        "diabetic person shaking pale sweating low blood sugar not making sense",
        "high blood sugar diabetic vomiting confused and drowsy need help",
        "diabetic emergency blood sugar very high feeling very sick vomiting",
        "my diabetic mother is confused lethargic blood sugar reading very high",
        "sugar crash diabetic shaking sweating confusion please help quickly",
    ]

    # ────────────────────────── MEDIUM (Label: medium) ──────────────────────────
    medium_transcripts = [
        # Fever and illness
        "i have a high fever for three days and a very bad cough feeling weak",
        "high temperature for two days body aches and sore throat feeling miserable",
        "fever of 103 degrees headache and body pain for two days not improving",
        "my child has a high fever he is vomiting and seems lethargic",
        "sick with flu high fever chills body aches for four days now",
        "severe cold symptoms bad cough runny nose fever body aches need help",
        "persistent fever for a week and bad cough losing weight doctor needed",
        "stomach flu vomiting and diarrhea for two days feeling dehydrated",
        "food poisoning vomiting for hours stomach cramps feeling very weak",
        "high fever and rash appeared today body aches concerned about infection",
        # Moderate pain
        "sprained my ankle from jogging its swollen and hurts to walk",
        "pulled a muscle in my back hurts to move but i can still walk",
        "wrist pain from typing too much swollen and aching could be carpal tunnel",
        "knee pain getting worse after hiking swollen and stiff",
        "shoulder pain from gym workout cant lift arm above head",
        "neck pain and stiffness for days cant turn head properly",
        "lower back pain radiating down leg making it hard to walk",
        "elbow pain from tennis swollen and sore for days",
        "hip pain making it hard to walk need to see doctor",
        "rib pain from coughing too hard hurts to breathe deeply",
        # Minor infections
        "infected wound on my hand red swollen warm to touch pus coming out",
        "ear infection severe ear pain and fever for two days",
        "urinary infection burning pain and frequent urination need antibiotics",
        "infected insect bite on leg red swollen spreading small red streaks",
        "sinus infection bad headache facial pressure and thick nasal discharge",
        "infected finger from hangnail swollen red throbbing with pus",
        "eye infection pink eye symptoms watery itchy and crusty",
        "throat infection strep throat symptoms white spots severe sore throat",
        "skin infection cellulitis spreading redness warmth and swelling on leg",
        "dental infection toothache worsening gum swollen some fever",
        # Allergic reactions mild-moderate
        "allergic reaction hives all over body from new medication itching badly",
        "allergic reaction to food face slightly swollen itchy hives",
        "rash spreading after using new soap very itchy and red",
        "mild allergic reaction to bee sting swelling at site itchy and red",
        "seasonal allergy attack sneezing watery eyes difficulty concentrating",
        "skin rash from allergy itchy bumps spreading across arms and legs",
        "allergic reaction to medicine mild swelling and rash appeared today",
        "contact dermatitis from plants both hands and arms red blistering itchy",
        "hay fever symptoms are terrible cant stop sneezing eyes swollen",
        "nickel allergy rash on wrist and neck from jewelry itchy and red",
        # Psychological distress moderate
        "having a panic attack i cant calm down my chest feels tight and i feel dizzy",
        "severe anxiety heart racing cant think straight feeling like something is wrong",
        "having a bad panic episode cant breathe properly feel like i am dying",
        "intense anxiety attack shaking sweating heart pounding feeling overwhelmed",
        "emotional crisis feel very anxious and depressed cant stop crying need help",
        "feeling very dizzy and lightheaded almost passed out need advice",
        "my blood pressure is high symptoms are headache and dizziness worried",
        "hypertension acting up headache blurry vision feeling unwell",
        "migraine very severe nauseous light sensitive worst headache ever",
        "cluster headache unbearable pain behind eye feels like i am being stabbed",
        # Mild breathing issues
        "mild asthma flare up chest slightly tight but inhaler is helping",
        "shortness of breath when climbing stairs getting worse recently",
        "occasionally wheezing after exercise not sure if it is asthma",
        "cough wont go away for weeks mild shortness of breath sometimes",
        "chest tightness after walking fast but goes away when i rest",
        # Dehydration / exhaustion
        "very dehydrated after workout dizzy and weak headache",
        "heat exhaustion feeling faint nauseous sweating a lot",
        "fatigued and weak for days barely eating or drinking",
        "dehydrated from stomach bug cant keep water down dizzy",
        "overheated and dizzy at outdoor event need water and rest",
        # Minor wounds needing care
        "scrape on knee from falling off scooter raw and bleeding a bit",
        "blister popped on foot from new shoes raw and painful",
        "paper cuts on finger wont stop bleeding more than expected",
        "splinter deep in palm cant get it out area getting red and sore",
        "toenail came off partially from stubbing toe bleeding and sore",
        # Pregnancy mild concerns
        "pregnant and having mild cramping and spotting worried about baby",
        "morning sickness is severe cant keep any food down losing weight",
        "pregnant experiencing braxton hicks contractions unsure if real labor",
        "pregnancy swelling in feet and hands headache feeling off",
        "back pain during pregnancy getting worse hard to sleep or move",
        # Elderly concerns
        "elderly father confused today not making sense but is awake and alert",
        "grandmother is dizzy and weak today nearly fell needs checkup",
        "elderly neighbor hasnt been eating well looking weak and pale",
        "senior citizen feeling tired and short of breath after light activity",
        "my elderly mother keeps forgetting things and seems confused today",
    ]

    # ────────────────────────── LOW (Label: low) ──────────────────────────
    low_transcripts = [
        "small blister on foot from walking too much in new shoes",
        "i cut my thumb while opening a box bleeding a tiny bit",
        "i scraped my knee falling over surface cut minor bleeding",
        "paper cut on paper bleeding very little applied pressure",
        "accidentally cut myself shaving its bleeding lightly",
        "minor cut from glass shard washed it not serious",
        "i cut my finger while chopping vegetables its bleeding a little but i washed it",
        "small paper cut on my hand hurts a bit but its not deep",
        "scraped my knee while running just a surface wound nothing serious",
        "got a small cut while shaving stopped bleeding already just stings",
        "i have a mild headache and wondering what medicine to take",
        "feeling a little tired today just need some advice on rest",
        "have a small cold with runny nose and sneezing nothing serious",
        "slight sore throat maybe beginnings of a cold need advice",
        "mild stomach ache probably ate something bad earlier today",
        "feeling slightly nauseous after lunch dont think its serious",
        "eyes are a bit tired from screen time any tips for relief",
        "have a mild cough nothing concerning just want to know remedies",
        "feeling a bit lightheaded probably just hungry or dehydrated",
        "mild muscle soreness from gym workout yesterday need stretching advice",
        # Non-medical / stress
        "hey i lost my keys and i am really stressed out can you help",
        "i am feeling stressed about work deadlines any relaxation tips",
        "having trouble sleeping for a couple nights not sure why",
        "feeling a bit anxious about an upcoming presentation normal stress",
        "i stubbed my toe and it hurts a bit but seems okay",
        "insect bite on arm small red bump itching a little bit",
        "minor sunburn on shoulders slightly red and warm not blistering",
        "dry skin on hands cracking slightly its winter and my hands are dry",
        "mosquito bite on leg itchy but otherwise fine want to know remedies",
        "small blister on foot from walking too much in new shoes",
        
        # Massive Dataset Expansion (Low - minor injuries with scary keywords context)
        "minor cut on my finger its bleeding a tiny bit but easily stopped",
        "little scrape on my knee just a surface wound nothing serious",
        "small paper cut barely bleeding just hurts a bit",
        "tiny scratch from my cat not deep at all",
        "mild headache taking some ibuprofen",
        "stubbed my toe really hard hurts but not broken",
        "just a slight sore throat probably allergies",
        "nothing serious just a small bruise on my arm",
        "a bit of a stomach ache after eating spicy food",
        "got a little splinter in my thumb needs tweezing",
        "minor burn from curling iron tiny blister",
        "small nick while shaving slight bleeding",
        "just feeling a little tired today need to rest",
        "mild sunburn on my shoulders nothing major",
        "scraped my elbow slightly falling on turf",
        "tiny cut on my thumb from a knife not deep",
        "minor scrape on shin from a pedal",
        "just a small scratch on my face",
        "little bit of blood from a picked scab",
        "mild muscle cramp in my calf from running",
        "small blister from new shoes slightly annoying",
        "minor papercut bleeding a tiny amount",
        "just a slight cough nothing else",
        "mild allergy symptoms sneezing a little",
        "small bump on the head no dizziness just a little sore",
        "minor scrape on hand washed it with soap",
        "little bite from a mosquito very itchy",
        "just a small rash on my wrist from a watch",
        "mild toothache nothing severe just annoying",
        "tiny piece of dust in eye washed it out",
        "small nick on ankle from a razor",
        "just a minor nosebleed stopped after 5 minutes",
        "little scratch on knee from a bush",
        "mild indigestion took an antacid",
        "small bruise on thigh from walking into table",
        "minor cut on lip from dry weather",
        "just a slight headache from lack of caffeine",
        "little bit of nausea probably car sickness",
        "small scratch on arm barely noticeable",
        "minor burn from hot coffee touching tongue",
        "just a mild strain in my shoulder",
        "tiny cut under fingernail annoying but fine",
        "small insect bite on neck slightly swollen",
        "minor scrape on palm of hand",
        "just a little dizzy from standing up too fast",
        "mild throat tickle maybe dry air",
        "small cut on toe from a rock",
        "little bruise on shin",
        "minor scrape on knee from tripping",
        "just a small cut nothing to worry about",
        "tiny scratch bleeding a very little bit",
        "mild headache and little bit of a sniffle",
        "small blister on heel from running",
        "minor burn from touching a hot plate briefly",
        "just slightly nauseous after a big meal",
        "little scratch from a rose bush thorn",
        "mild back ache from sitting too long",
        "small cut on finger stopped bleeding quickly",
        "minor scrape on elbow put a bandaid on it",
        "just a tiny splinter not deep",
        "little bit of a sore throat",
        "mild sunburn on arms",
        "small bruise on arm",
        "minor cut on knuckle",
        "just a slight stomach bug",
        "little scratch on leg",
        
        # Wellness questions
        "want to know the best first aid kit items o keep at home",
        "asking about general health tips for staying well during flu season",
        "what should i do if i get a bee sting just general advice",
        "how do i take care of a minor burn from cooking",
        "what are signs of dehydration and how can i prevent it",
        "best way to treat a mild sprain at home need basic advice",
        "can you recommend some stretches for mild back pain",
        "what should i eat when i have a mild stomach bug",
        "how do i clean a minor wound properly at home",
        "is it normal to feel tired after sleeping too much",
        # Very minor issues
        "got a hangnail its a bit sore but nothing major",
        "my nose is runny and i have been sneezing all day just a cold",
        "mild indigestion after a big meal feel bloated and gassy",
        "slight earache probably from swimming yesterday not too bad",
        "chapped lips from cold weather looking for good treatment options",
        "dry eyes from computer work what eye drops should i use",
        "mild hay fever symptoms sneezing a little bit runny nose",
        "minor bruise on shin from bumping into table barely hurts",
        "have a canker sore in my mouth its annoying but not painful",
        "itchy scalp probably dandruff nothing medical just want advice",
        # Followup / non-urgent
        "following up on a doctors appointment just had blood tests done",
        "need to refill my prescription for blood pressure medication",
        "wondering when i should schedule my next flu vaccine",
        "had a checkup last week everything seems fine just confirming",
        "minor rash went away on its own just documenting it",
        "feeling much better after being sick last week just checking in",
        "recently got over a cold cough still lingers slightly",
        "wondering about side effects of a new vitamin i started taking",
        "had minor procedure done healing well no complications noticed",
        "asking about recommended daily water intake for my weight",
        # Random non-emergency
        "just wanted to know if i should ice or heat a sore muscle",
        "whats the best way to bandage a blister on my heel",
        "how long should i rest after a mild ankle twist",
        "is it okay to exercise with a mild cold or should i rest",
        "can i take two different cold medicines at the same time",
        "what is the normal heart rate for someone my age 30 years",
        "should i be worried about occasional heartburn after meals",
        "how do i remove a tick properly without leaving the head",
        "whats the best antiseptic for cleaning minor wounds at home",
        "is it safe to pop a small blister or should i leave it alone",
        # Truly benign
        "i am just bored and wanted to chat about health topics",
        "what are some good healthy recipes for meal prep",
        "can you suggest a good workout routine for beginners",
        "how many hours of sleep should i get per night",
        "what vitamins should a 25 year old be taking daily",
        "tips for improving posture while working at a desk",
        "how do i start meditating for stress relief",
        "what foods are good for immune system support",
        "best ways to stay hydrated during summer heat",
        "general advice on maintaining mental health and wellness",
    ]

    # Assign labels
    for t in critical_transcripts:
        data.append((t, 'critical'))
    for t in high_transcripts:
        data.append((t, 'high'))
    for t in medium_transcripts:
        data.append((t, 'medium'))
    for t in low_transcripts:
        data.append((t, 'low'))

    random.shuffle(data)
    return data


# ==================== TRAINING PIPELINE ====================
def train_models():
    print('=' * 80)
    print('🧬 NARAYANI SEVERITY MODEL TRAINING PIPELINE v4')
    print('   Enterprise-Grade Emergency Severity Classifier')
    print('=' * 80)

    # 1. Generate data
    print('\n📦 Step 1: Generating training dataset...')
    data = generate_training_data()
    transcripts = [d[0] for d in data]
    labels = [d[1] for d in data]

    class_counts = {}
    for l in labels:
        class_counts[l] = class_counts.get(l, 0) + 1
    print(f'   Total samples: {len(data)}')
    for cls, cnt in sorted(class_counts.items()):
        print(f'   {cls:>10}: {cnt} samples')

    # 2. TF-IDF
    print('\n🔤 Step 2: Training TF-IDF Vectorizer...')
    tfidf = TfidfVectorizer(
        max_features=300,
        ngram_range=(1, 3),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        strip_accents='unicode',
    )
    tfidf_matrix = tfidf.fit_transform(transcripts)
    print(f'   TF-IDF features: {tfidf_matrix.shape[1]}')

    # 3. Keyword features
    print('\n🔑 Step 3: Extracting keyword features...')
    keyword_features = [extract_features(t) for t in transcripts]
    keyword_sparse = csr_matrix(keyword_features)
    print(f'   Keyword features: {keyword_sparse.shape[1]}')

    # 4. Combine
    print('\n🔗 Step 4: Combining feature matrices...')
    combined = hstack([tfidf_matrix, keyword_sparse])
    print(f'   Combined features: {combined.shape[1]}')

    # 5. Scale
    print('\n⚖️  Step 5: Scaling features...')
    scaler = MaxAbsScaler()
    X = scaler.fit_transform(combined)

    # 6. Encode labels
    print('\n🏷️  Step 6: Encoding labels...')
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    print(f'   Classes: {list(encoder.classes_)}')

    # 7. Train individual models
    print('\n🏋️  Step 7: Training individual models...\n')

    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=25,
        min_samples_split=3,
        min_samples_leaf=1,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
    )
    print('   Training Random Forest (500 trees)...')
    t0 = time.time()
    rf.fit(X, y)
    print(f'   ✅ RF trained in {time.time() - t0:.1f}s')

    xgb = XGBClassifier(
        n_estimators=500,
        max_depth=10,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        use_label_encoder=False,
        eval_metric='mlogloss',
        n_jobs=-1,
    )
    print('   Training XGBoost (500 rounds)...')
    t0 = time.time()
    xgb.fit(X, y)
    print(f'   ✅ XGB trained in {time.time() - t0:.1f}s')

    lgbm = LGBMClassifier(
        n_estimators=500,
        max_depth=15,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    print('   Training LightGBM (500 rounds)...')
    t0 = time.time()
    lgbm.fit(X, y)
    print(f'   ✅ LGBM trained in {time.time() - t0:.1f}s')

    # 8. Ensemble
    print('\n🤝 Step 8: Building Voting Ensemble...')
    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('xgb', xgb), ('lgbm', lgbm)],
        voting='soft',
        n_jobs=-1,
    )
    t0 = time.time()
    ensemble.fit(X, y)
    print(f'   ✅ Ensemble trained in {time.time() - t0:.1f}s')

    # 9. Cross-validate
    print('\n📊 Step 9: Cross-validation (5-fold)...\n')
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in [('RF', rf), ('XGB', xgb), ('LGBM', lgbm), ('Ensemble', ensemble)]:
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
        print(f'   {name:>10}: {scores.mean():.4f} ± {scores.std():.4f}')

    # 10. Full classification report
    print('\n📋 Step 10: Full classification report on training set...\n')
    y_pred = ensemble.predict(X)
    print(classification_report(y, y_pred, target_names=encoder.classes_))

    # 11. Save models
    print('\n💾 Step 11: Saving all 7 model artifacts...')
    artifacts = {
        'narayani_ensemble_v3.pkl': ensemble,
        'narayani_encoder_v3.pkl': encoder,
        'narayani_scaler_v3.pkl': scaler,
        'narayani_tfidf_v3.pkl': tfidf,
        'narayani_rf_v3.pkl': rf,
        'narayani_xgb_v3.pkl': xgb,
        'narayani_lgbm_v3.pkl': lgbm,
    }

    for fname, model in artifacts.items():
        path = os.path.join(SAVE_DIR, fname)
        joblib.dump(model, path)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f'   ✅ {fname} ({size_mb:.1f} MB)')

    print('\n' + '=' * 80)
    print('🎉 TRAINING COMPLETE — All v3 models saved to Datasets/')
    print('   Restart severity_service.py to pick up the new models.')
    print('=' * 80)


if __name__ == '__main__':
    train_models()
