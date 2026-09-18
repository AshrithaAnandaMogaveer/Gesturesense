# ISL Gesture Recognition Guide

## What Can The System Recognize?

The system recognizes **114 Indian Sign Language (ISL) WORDS** - these are full word gestures, not individual alphabet letters.

---

## 114 ISL Words You Can Show

### Common Greetings & Polite Words
- **HELLO_HI** - Open palm wave or flat hand forward
- **THANK** - Hand starts at lips/chin, moves forward
- **SORRY** - Fist circling on chest area
- **PLEASE** - Open hand circling on chest
- **WELCOME** - Both hands sweeping inward

### Questions Words
- **WHAT** - Hands spread, questioning expression
- **WHERE** - Index finger pointing, questioning
- **WHO** - Index finger pointing at people
- **HOW** - Hands showing "how" gesture
- **WHEN** - (Time-related gesture)

### Common Actions
- **GO** - Hand/finger pointing forward/away
- **COME** - Hand beckoning toward self
- **DO** - Hands showing action
- **HELP** - One hand supporting the other
- **BRING** - Hands bringing something close
- **LEAVE** - Hand pushing away
- **STOP** - Flat hand raised (stop sign)
- **TAKE CARE** - Protective gesture
- **TURN ON** - Twisting/switching motion

### Daily Needs
- **FOOD** - Hand to mouth (eating motion)
- **WATER** - Cupped hand to mouth (drinking)
- **SLEEP** - Hands together by head (sleeping)
- **MEDICINE** - Taking medicine gesture
- **PHONE** - Hand by ear (calling)
- **BED** - Head resting on hands
- **ROOM** - Outlining a space
- **SITTING** - Sitting position gesture
- **WEAR** - Putting on clothes motion

### Emotions & Feelings
- **HAPPY** - Smiling, hands up joyfully
- **ANGRY** - Tense expression, fists
- **SAD** - (Equivalent: CRYING, DISAPPOINTED)
- **TIRED** - Exhausted expression, drooping
- **BORED** - Uninterested expression
- **AFRAID** - Protective, fearful gesture
- **CRYING** - Tears motion on cheeks
- **WORRIED** - (WORRY) - Concerned expression
- **DISAPPOINTED** - Let down gesture
- **GRATEFUL** - Appreciative gesture

### Physical States
- **HUNGRY** - Hand on stomach
- **THIRSTY** - Throat gesture
- **COLD** - Shivering/hugging self
- **FEVER** - Hand on forehead (hot)
- **HURT** - Pain indication
- **FINE** - Okay gesture

### People & Relationships
- **I_ME_MINE_MY** - Pointing to self
- **YOU** - Pointing to other person
- **FRIEND** - Linked fingers or hands together
- **SOME ONE** - Indicating a person

### Communication
- **TALK** - Fingers opening/closing (speaking)
- **CHAT** - Conversation gesture
- **SPEAK** - Mouth/voice gesture
- **TELL** - Informing gesture
- **HEAR** - Hand by ear (listening)
- **REPEAT** - Circular motion (say again)

### Positive Actions
- **LIKE** - Thumbs up or positive gesture
- **LIKE_LOVE** - Heart gesture or affection
- **AGREE** - Nodding or agreement sign
- **ENJOY** - Happy, enjoying gesture
- **APPRECIATE** - Gratitude gesture
- **CONGRATULATIONS** - Celebration gesture
- **SURE** - Certainty gesture
- **PROMISE** - Pinky promise or oath gesture
- **TRUST** - Confidence gesture

### Negative/Refusal
- **NOT** - Shaking head/hand
- **DON'T CARE** - Dismissive gesture
- **ABUSE** - Negative aggressive gesture
- **STUBBORN** - Resistant gesture

### Descriptive Words
- **GOOD** - Thumbs up or positive
- **BAD** - Thumbs down or negative
- **NICE** - Pleasant gesture
- **BEAUTIFUL** - Attractive gesture
- **KIND** - Gentle, caring gesture
- **FREE** - Released, available gesture
- **REALLY** - Emphasis gesture
- **SO MUCH** - Large amount gesture
- **A LOT** - Many/much gesture
- **SOFTLY** - Gentle motion
- **SLOWER** - Slow down motion

### Places & Locations
- **COLLEGE_SCHOOL** - Education building gesture
- **OUTSIDE** - Outward gesture
- **PLACE** - Location indication
- **FROM** - Origin/source gesture

### Time & Frequency
- **TODAY** - Current day gesture
- **ON THE WAY** - Coming/traveling
- **TAKE TIME** - Duration gesture

### Abstract Concepts
- **DIFFERENCE** - Contrasting gesture
- **DILEMMA** - Confusion/choice gesture
- **HEART** - Chest/emotion area
- **TRUTH** - Honesty gesture
- **MEAN IT** - Sincerity gesture
- **THINGS** - Items/objects
- **NUMBER** - Counting/quantity
- **CLASS** - Group/category
- **OLD_AGE** - Elderly/age gesture

### Actions & Activities
- **PREPARE** - Getting ready
- **MEET** - Meeting gesture
- **SERVE** - Serving/helping
- **POUR** - Pouring liquid motion
- **COMB** - Combing hair motion
- **HIDING** - Concealing gesture
- **TRAIN** - Training/learning

### Additional Words
- **ALL** - Everything, complete
- **ANYTHING** - Any item
- **SOMETHING** - Some item
- **SOME HOW** - In some way
- **NAME** - Naming/identity
- **FAVOUR** - Request/help
- **DARE** - Challenge gesture
- **HAD** - Past possession
- **HAPPENED** - Past event
- **BECOME** - Transformation
- **THINK** - Thinking gesture (head)
- **UNDERSTAND** - Comprehension gesture
- **WANT** - Desire gesture
- **NEED** - (Similar to WANT)

---

## Important Notes

### ⚠️ What This System Does NOT Recognize

❌ **Individual alphabet letters** (A, B, C, D, etc.)  
❌ **Numbers** (1, 2, 3, 4, etc.)  
❌ **Generic gestures** (Thumbs up, Peace sign, OK sign)  
❌ **Random hand shapes** - only trained ISL words

### ✅ What This System DOES Recognize

✅ **114 ISL words** (full words, not letters)  
✅ **Common conversation words** (greetings, questions, actions)  
✅ **Real ISL gestures** from the ISL_CSLRT_Corpus dataset  

---

## How to Use the System

### 1. **Check the Dataset Images**
The best way to know what gestures to show is to look at the original training images:

```
Location: ISL_CSLRT_Corpus/ISL_CSLRT_Corpus/Frames_Word_Level/
```

Each folder contains example images of that specific ISL word gesture.

### 2. **Tips for Good Recognition**

✅ **Good lighting** - Clear, bright environment  
✅ **Hand in frame** - Center your hand in the camera view  
✅ **Steady gesture** - Hold the gesture for 0.3-0.5 seconds  
✅ **Clear hand shape** - Make the gesture distinct and clear  
✅ **Right distance** - Not too close or too far (arm's length)  
✅ **Use actual ISL gestures** - Match the training data gestures  

### 3. **Understanding the Output**

The system shows:
- **Gesture name** (e.g., "HELLO_HI", "THANK", "FOOD")
- **Confidence percentage** (e.g., 68%, 45%, 32%)

**Confidence Guide:**
- **>60%** - Very confident (high accuracy)
- **40-60%** - Confident (good accuracy)
- **30-40%** - Uncertain (may be correct)
- **<30%** - Very uncertain (likely incorrect)

---

## Quick Start Guide

### Easy Gestures to Try First:

1. **I_ME_MINE_MY** - Point to yourself
   - This is the easiest! Just point at your chest.

2. **YOU** - Point to the camera/other person
   - Point forward at the camera.

3. **FOOD** - Hand to mouth (eating motion)
   - Bring fingers to your mouth (eating gesture).

4. **WATER** - Cupped hand to mouth (drinking)
   - Make a "C" shape and bring to mouth (drinking).

5. **HELLO_HI** - Open palm wave
   - Show open palm or wave hand.

6. **THANK** - Hand from lips/chin forward
   - Start at chin/lips, move hand forward.

7. **HELP** - One hand on other palm
   - Place one hand on the other palm (supporting).

### Medium Difficulty:

8. **HAPPY** - Smiling with hands up
9. **SORRY** - Fist circling on chest
10. **GOOD** - Positive gesture (thumbs up position)
11. **STOP** - Flat hand raised (stop sign)
12. **GO** - Pointing forward/away

### Check the Training Images:

For the **exact gesture**, look at the images in:
```
ISL_CSLRT_Corpus/ISL_CSLRT_Corpus/Frames_Word_Level/[WORD_NAME]/
```

Each folder has 2-100+ example images showing how to perform that gesture.

---

## Complete List of 114 Words

```
A LOT, ABUSE, AFRAID, AGREE, ALL, ANGRY, ANYTHING, APPRECIATE, BAD, BEAUTIFUL,
BECOME, BED, BORED, BRING, CHAT, CLASS, COLD, COLLEGE_SCHOOL, COMB, COME,
CONGRATULATIONS, CRYING, DARE, DIFFERENCE, DILEMMA, DISAPPOINTED, DO, DON'T CARE,
ENJOY, FAVOUR, FEVER, FINE, FOOD, FREE, FRIEND, FROM, GO, GOOD, GRATEFUL, HAD,
HAPPENED, HAPPY, HEAR, HEART, HELLO_HI, HELP, HIDING, HOW, HUNGRY, HURT,
I_ME_MINE_MY, KIND, LEAVE, LIKE, LIKE_LOVE, MEAN IT, MEDICINE, MEET, NAME,
NICE, NOT, NUMBER, OLD_AGE, ON THE WAY, OUTSIDE, PHONE, PLACE, PLEASE, POUR,
PREPARE, PROMISE, REALLY, REPEAT, ROOM, SERVE, SHIRT, SITTING, SLEEP, SLOWER,
SO MUCH, SOFTLY, SOME HOW, SOME ONE, SOMETHING, SORRY, SPEAK, STOP, STUBBORN,
SURE, TAKE CARE, TAKE TIME, TALK, TELL, THANK, THAT, THINGS, THINK, THIRSTY,
TIRED, TODAY, TRAIN, TRUST, TRUTH, TURN ON, UNDERSTAND, WANT, WATER, WEAR,
WELCOME, WHAT, WHERE, WHO, WORRY, YOU
```

---

## Troubleshooting

### "Unknown Gesture" appears:
- Your gesture doesn't match any of the 114 trained words
- Try one of the common gestures listed above
- Check the training images for reference
- Ensure good lighting and clear hand position

### Low confidence (<40%):
- Gesture may be unclear or ambiguous
- Try holding the gesture more steadily
- Improve lighting conditions
- Make sure hand is fully visible
- Check if you're doing the exact ISL gesture

### Multiple predictions alternating:
- Similar gestures (e.g., THANK ↔ THAT)
- Hold gesture steady for 0.5 seconds
- Make the gesture more distinct
- Check training images for differences

---

## Summary

**You can show:** 114 ISL word gestures (not alphabet letters)  
**Best source:** Check images in `ISL_CSLRT_Corpus/Frames_Word_Level/[WORD]/`  
**Start with:** I_ME_MINE_MY, YOU, FOOD, WATER, HELLO_HI, THANK, HELP  
**Result:** System shows word name + confidence %  

**Test at:** http://127.0.0.1:5000 🚀
