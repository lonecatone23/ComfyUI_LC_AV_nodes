# H3 Showrunner Prompting Guide 🎬

How to fill in the cards on **LC H3 Story** so MiniMax H3 makes the movie you meant. Everything here comes from test renders (the Coffee Date movie and a wardrobe test with three people, two outfits, two props, inserted pictures and a 360 room), not theory.

---

## ✋ The three rules that matter most

- **Words beat pictures.** H3 and Krea 2 both follow the words first. A picture helps, but when the words and the picture disagree, the words win. Write it down even when the picture shows it.
- **One thing, one name, everywhere.** A person, an outfit or a prop is found by its exact name. "the red dress" on the card and "in the red dress" in the action match. "her red dress" doesn't.
- **Every card picture is copied, not just its face.** The background, the clothes, the pose and the expression all leak into the shots. That's why cards are plain: neutral face, plain grey background, plain clothes.

---

## How H3 reads your cards

You never write a prompt by hand. The Story node builds one per shot, and that text is **exactly** what H3 gets (open a shot's **H3 prompt** to read it; **↻** refreshes it, **✎ Edit** lets you rewrite it):

- **The cast line**: each person's name, who they are, their look, and "wearing <outfit>" when they're in one. Plus the movie's look.
- **Place:** the place card's description.
- **Camera:** the framing, who it's on, the move, and what is behind them.
- **Timed beats:** the action, the line and the after, each at its moment ("At 0.60 seconds, Nova says: "..."").
- **Sound:** the place's sound, plus "The only voice is the quoted line" or "Nobody speaks".
- With it go the pictures the shot names: faces (and voices), the outfit each person wears now, the props it mentions, the room or 360 angle, and the start frame.

---

## 🎭 Cast

### The look
- **One line, one clause:** age, hair, face. "in her twenties, long wavy blonde hair, light freckles, green eyes". A period in the middle splits it, and half can land on the wrong person.
- **Hair length, always.** "Blonde hair" gets whatever length H3 picks this time. "Long wavy blonde hair" stays.
- **Something different for each person.** Two women with similar hair blend. Change the hair color or length, or add a detail ("small silver nose ring").
- **Names:** a capital letter, short, different from each other.
- **who** (woman, man, girl, boy, person, cat, dog) is how H3 tells people apart. Set it.
- **Clothes go on outfit cards** (below). Clothes in the look work too, but then they're on that person in every shot.

### The cast card picture
- **A front-facing headshot on plain grey.** It gives Krea 2 Identity Edit the face and almost no clothes to copy. The Story Card Prompt (kind **cast card**) makes exactly that.
    - Tested: a torso card's shirt went into every shot of whoever had no outfit, and a torso card with no clothes in the look came out bare. Headshots gave the same face match with neither problem.
- **Neutral face.** A smiling card gives you a smiling actor in every shot. It pulls, it's not a hard lock.
- **Your own photo works**, even seated or turned a little (tested). Drop it on the card: the background comes off and it goes on grey. A sharp, well-lit face matters more than the pose.

### ✋ Everyone needs clothes from somewhere
- A headshot card carries no clothes. **Give every person an outfit card**, or clothing words in their look.
- With neither, H3 and Krea dress them differently in every shot (tested: a black sweater, a grey tee, a black top and grey knit in four shots). The Story warns you (👗).

---

## 👗 Outfits

An outfit card dresses one person from the shot that names it on. The cast line gets "wearing <outfit>", the shot gets the outfit picture, and the storyboard frame shows it.

### ✋ A color for every item
- **Every piece of clothing gets its own color.** "light-wash denim jacket over a plain white t-shirt, black skinny jeans", not "a jacket and jeans". An item with no color gets one picked for you, and it won't be the same twice.
- **The shade, not just the color:** light, dark, faded, muted, deep, soft, pale. "faded olive", "deep red", "dusty-blue", "warm terracotta", "cream". Plain "blue" drifts.
- **Layers outside first, with "over":** "denim jacket over a white t-shirt".
- **Patterns name their colors:** "floral-print blouse in pale peach and green".
- **The fabric helps:** knit, linen, silk, denim, leather, wool.
- **Shoes and accessories** only when they'd show, with a color.

### The outfit card
- **name**: short, the words you'll use in the action: "the red dress", "the denim outfit".
- **who**: the person who wears it. Empty = nobody does (a ⚠️ says so).
- **description**: the whole outfit with every color: "a deep red silk slip dress with thin straps".
- **picture**: kind **outfit** makes it on an invisible mannequin, on grey. Your own picture works (the background comes off).

### How it plays
- **Name the first outfit in the person's first shot** ("Nova, in the denim outfit, and Anika sit down."). Before an outfit is named, the person has no clothes from anywhere.
- The shot that names another one ("Nova comes back to the table in the red dress") switches them. **They stay in it** until a shot names another of their outfits.
- To change back, name the first outfit again.
- ✅ Tested on video: two outfits on one person, the change held across the cut, and the other person's clothes never picked up either outfit.
- A close-up or medium shot only gets its own person's pictures, outfit included, so one person's outfit never lands on the person across the table.

---

## 📍 Places

- **description**: what is there, as things you could point at, with left and right: "a small French cafe with a marble counter on the left, big arched windows on the right, bentwood chairs, warm lamps".
- **sound**: sounds you can hear, nothing else: "rain on the glass, cups clinking, footsteps". Never write who speaks or "nobody speaks / no voices": naming speech makes H3 speak (tested 2026-11-03: the same silent shot with "nobody speaks except the quoted lines" in the sound talked in French and English on two seeds; without it, silence). "People talking" does the same.
- **room picture**: **empty**. A person in the room picture shows up in the movie.
- **🌐 360 panorama**: the whole room, horizon level, nobody in it. Make a few, keep the best. Then turn the place viewer to what your wide shot shows (**📷 the wide shot faces**) once.
    - Every close-up gets the right part of the room behind it, by itself: a close-up with no angle of its own gets the reverse of the person across from it.
    - ✅ Tested: close-ups from the 360 kept the same café behind each person.

---

## 👜 Props

- A prop card is for an object that must look the same every time (the umbrella, the notebook, the ring). Cups and chairs don't need one.
- ✋ **The action names it with the card's exact words.** Card "the yellow umbrella" + "The yellow umbrella leans against the table." = the shot gets the picture.
- **description** with colors: "a bright mustard-yellow umbrella with a curved black wooden handle".
- **picture**: kind **prop** (on grey), or your own photo: the background comes off. ✅ Tested with a photo of a notebook on a café table.
- **Count once:** "Each of them holds one coffee cup." Say "a cup" twice and you may get two.
- **Box it** (📐) in the first wide shot it appears in, or the storyboard can't place it and later shots invent it.

---

## 🎞 Shots

### Framing
- **wide shot**: the whole room and everyone. Open every scene with one: it becomes the place's anchor, and later wides there are built on it.
- **side view / two-shot**: two people.
- **medium shot**: one person, waist up. **close-up**: face and shoulders. **extreme close-up**: the face.
- **over the shoulder**, **from behind**, **custom** (you write the camera in the action).
- ✋ **A close-up or medium shot is one person: the one in "who it's on".** Only that person's pictures go with it.
- 🛑 **Don't name the other person in a close-up's action.** "Anika stares at Nova" made H3 cut to Nova halfway through Anika's own close-up, even with only Anika's pictures sent. "Anika stares across the table, surprised." kept her the whole shot (face match 0.32 → 0.79). The Story warns you (⚠️).

### Move
- **still** is the safest. One slow move per shot at most.

### ✂ Cut or 🎞 keep rolling
- **Cut** for almost everything. It gets the faces, the outfits, the props, the room and the anchor.
- **Keep rolling** only for the same camera a moment later: same framing, same place, same person on a close-up. It gets no pictures, just the last 1.6 s of the shot before.
- ⚠️ Change the framing on a keep-rolling shot and H3 cuts by itself to a room it invents. The Story warns you and makes it a cut.

---

## ✍️ Actions

- **Present tense, short sentences, one thing each.** "Nova sits down. She puts her bag on the table. She looks at Anika."
- **Small movements.** Gestures, glances, sitting, standing, a few steps.
- **Say what happens, not what doesn't.** "Nobody drinks" still had her drinking. Write "Nova rests her hands on the table."
- **Name people first, "she" after.**
- **Never invent people.** A waiter with no cast card is a stranger with a random face.
- **Timed beats:** `At 0.00 seconds, Nova looks up. At 1.20 seconds, she smiles.` uses your times exactly.

---

## 🗣️ Lines

- **Word for word** in the **line** box, and pick the **speaker**.
- **One line per shot.** Two people talking = shot and reverse shot.
- **About 2.5 words a second.** Auto length fits it. A typed length that's too short gets stretched.
- **No mouth actions during the line** (drinking, eating, smoking). The Story puts the cup down for the line; write the sip in **after**.
- **after**: what happens once the line is said. Empty = a silent beat, because H3 fills spare time by saying the line again.
- **Don't reorder the cast mid-project.** Speaker 1 is the first person on the cast list for the whole movie.

---

## 🔇 Silent shots

- A shot with no line is silent only if nothing in it mentions speech. The Story adds "Nobody speaks." itself: do not add your own.
- The cast voice is sent only to shots with a line.
- No mouth "slightly open" or "about to speak" wording.
- Stop a shot before a person reaches someone they could greet: a walk-up to the counter invited speech.
- If you hear a language you did not write, it is invented speech: look at the sound line first.

---

## 🎥 Close-ups across a table

- The camera words the Story writes for you: "faces the camera, eyes just past the lens to the left/right, toward the person across the table". The person on the left of the wide looks just right of the lens, and the other way round. That's what makes them look at each other across the cut.
- Say what's behind each of them in the shot's **behind** box (the 360 angle shows it, the words make it stick).
- "nobody in the foreground" in the action stops a shoulder from the other person creeping in at the edge.

---

## 🖼 Your own start frame (inserted layouts)

- Drop any picture on a shot's **🖼 start** slot and H3 starts the shot from it, exactly.
- ✋ **It has to be the same room as the shot's place.** Tested: a start frame of a different café opened the shot, then H3 cut to the place card's café a second later. The place picture, its anchor and the description all said "the other café".
    - Using a new room? Make it its own place card (with that picture as the room picture) and set the shot's **where** to it. ✅ Tested: the same picture then held for the whole shot, and Marco walked into it.
- The people in it must look like the cast cards. A stranger in the start frame stays a stranger.
- A keep-rolling shot ignores its start frame (it carries on from the shot before). Make it a cut.

---

## 🗺️ Storyboard boxes (📐)

- One box per person where they stand in the first frame, and pick who it is. A prop box takes the prop's name.
- Bigger box = closer. Low in the frame = in front.
- ✋ **A later wide in the same place is built on the first wide's frame, and only the boxes are redrawn.** Box only what **changes** (Nova coming back in the red dress). Box everyone and everyone is redrawn, and faces drift (tested: Anika's face changed when she was boxed for no reason).
- ✋ **The start frame is the action's first moment.** "Nova comes back to the table and sits down" with Nova already seated in the frame gave a second Nova walking in. Box her standing where she comes from, or write the action from where the frame is ("Nova sits back in her chair.").
- H3 copies the start frame exactly. Check every face on the board before you Make. ↻ the ones that are wrong.

---

## 🔧 When something goes wrong

- **A stranger's face in a shot:** the person has no cast card picture, or the line names someone with no card. Make the card.
- **The face on the storyboard isn't your cast:** ↻ it, and check the card is a clear front torso shot.
- **A close-up turns into the other person halfway:** the action names them. Write "across the table" instead.
- **A second copy of someone walks in:** the start frame already shows them where the action has them arriving. Match the frame to the first moment.
- **Two copies of a person in a close-up:** "who it's on" is "everyone", or the scene picture already had a person in it. Set "who it's on" to one person.
- **The wrong person talks:** the speaker box, or the cast was reordered after shots were made.
- **The room changed after a cut:** no anchor yet. Make the wide shot first, or 📌 a good frame from it.
- **Two cups:** counted twice. "Each holds one cup."
- **Clothes change color between shots:** an item without a color, or a plain one. Give every item a color with a shade.
- **Wrong clothes after a change:** the action didn't use the outfit's exact name, or its **who** is empty.
- **A person's face drifts in a later wide:** they were boxed but don't change. Box only what changes.
- **Someone's clothes change every shot:** they have no outfit card and no clothes in the look (👗 warning).
- **The shot opens on your picture, then jumps to another room:** the inserted start frame isn't the shot's place. Make it its own place.
- **Too many pictures in one shot (⚠️ 5 pictures):** every named person, outfit and prop adds one. Name fewer props in a busy wide.
- **Garbled speech:** the line is too long for the shot, or a mouth action during it.
- **Speech in a silent shot (any language):** speech words in the place's sound line. Sounds only.
- **The cup / handover comes before the line:** it is in the action, which plays first. Put it in the **after** box.
- **A person freezes at the end of the shot:** the shot is pinned to an end frame and they got there early. End the action on the move into that frame, and keep the shot short.
- **Faces and mouths garbled in a wide shot:** a face there is 45-75 px and H3 draws in 16 px blocks. Use a medium, two-shot or over-the-shoulder shot.
- **The person seen from behind wears someone else's clothes:** only the person the shot is on has their pictures sent. Put the other person's clothes in the cast card's outfit notes (the over-the-shoulder shot says them in words).
- **Leaving looks like arriving:** both shots use the same angle of the same place. Pick another 🌐 angle for the exit.
- **Every angle looks like daytime:** the 360 panorama was made in daylight. Make it again with the same time of day as the story.
- **The door or room is different after a cut:** H3 does not keep two cuts consistent. Keep cuts few, pin a start frame, or hide the cut in a keep-rolling shot.
- **The line is said twice:** spare time after the line. Add an **after** beat.
- **A keep-rolling shot jumps to a new room:** the framing changed. Make it a cut.
- **face match under 0.40:** ↻ the shot. If it keeps failing, it's the card picture or the look words.

---

*"True Nothing is. Permitted Everything is"*- Yoda Auditore, **Assassin's Wars**
