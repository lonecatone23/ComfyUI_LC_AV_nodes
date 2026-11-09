# Lonecat's H3 Showrunner 🎬: Prompting Guide

How to fill in the cards on **LC H3 Story** so MiniMax H3 makes the movie you meant. Everything here comes from test renders (Coffee Date, Uber run, a wardrobe test with three people, outfits, props, inserted pictures and 360 rooms), not theory.

For what every button does, see the [User Guide](USER_GUIDE.md).

---

## ✋ The four rules that matter most

- **Words beat pictures.** H3 follows the words first. A picture helps, but when the words and the picture disagree, the words win.
    - Tested: an outfit card named "blue dress" with a picture of a red dress came out blue, 8 of 8 times.
- **One thing, one name, everywhere.** A person, an outfit or a prop is found by its exact name. "the red umbrella" on the card and "opens the red umbrella" in the action match. "her umbrella" doesn't.
- **Every card picture is copied, not just its face.** The background, the clothes, the pose and the expression all leak into the shots. That's why cards are plain: neutral face, plain grey, head and hair only on a headshot.
- **Show it, don't describe it.** MiniMax H3 is a reference model. An empty room, grey cast cards and a sharp start frame do more than any amount of wording.

---

## How H3 reads your cards

You never write a prompt by hand. The Story builds each shot's lines (open a shot's **H3 prompt**; **↻** refreshes them, **✎ Edit** lets you rewrite them), and Make wraps them into the full prompt when the shot is made. Together, H3 gets:

- **The cast line:** each person in the shot, who they are, their look, and "wearing <outfit>". Plus the movie's look. (An empty shot gets "No people are in this shot" instead.)
- **Place:** the place card's description.
- **Camera:** the framing, who it's on, the move, and what is behind them.
- **Timed beats:** the action, the line and the after, each at its moment ("At 0.60 seconds, Nova says: "..."").
- **Sound:** the place's sound, plus "The only voice is the quoted line" or "Nobody speaks."
- With it go the pictures the shot needs: faces (and voices), the outfit each person wears now, the props it names, the room or 360 angle, and the start frame.

---

## 🎭 Cast

### The look
- **One line, head and hair only:** age, face shape, eyes, freckles, skin, hair and its length. "mid twenties, a soft oval face, green eyes, light freckles, long wavy blonde hair".
- **Hair length, always.** "Blonde hair" gets whatever length H3 picks this time. "Long wavy blonde hair" stays.
- 🛑 **No clothes.** Clothes in the look beat the outfit picture and pull close-ups wide (H3 backs the camera off to fit the boots). The Story warns you (👗).
- 🛑 **No body words** (breasts, curvy, figure, body). Next to a bare-shouldered headshot they can make H3 leave the clothes off (lol, no. you do that under outfits). The outfit picture shows the body. The Story warns you (👗).
- **Something different for each person.** Two women with similar hair blend. Change the hair color or length, or add a detail ("small silver nose ring").
- **Proper Names:** a capital letter, one word, different from each other. "Parka girl" reads as two words: use "Anika".
- **who** (woman, man, girl, boy, person, cat, dog) is how H3 tells people apart. Set it.

### The headshot
- **Head and hair, on plain grey, neutral face.** No shoulders, no clothes: clothes in a headshot beat every outfit picture.
- **Neutral face.** A smiling card gives you a smiling actor in every shot. It pulls, it's not a hard lock.
- **Your own photo works**, even seated or turned a little. Drop it on the card: the background comes off and it goes on grey. ✂ crop it to the head. A sharp, well-lit face matters more than the pose.
- ⚠️ **Then rewrite the look to describe your picture.** ✨ makes a headshot from the look, so they match. Your own picture keeps whatever look was there, and the words win: a blonde headshot with "short dark hair" in the look comes out short and dark in every shot.

### ✋ Everyone needs an outfit from somewhere
- The headshot carries none. Give every person a **👗 outfit picture** on their cast card: that's what they wear from their first shot.
- With none, they get different clothes in every shot (tested: a black sweater, a grey tee, a black top and grey knit in four shots). The Story warns you (👗).

---

## 👗 Outfits

### The cast card's own outfit
- **outfit notes:** the whole outfit, head to toe, a color for every item. The 👗 picture is made from it.
- ✋ **Start with the main piece and its color.** H3 reads the first piece of clothing as the outfit's name: "dark blue jumpsuit, bandoleer, knee high boots" is sent as "wearing dark blue jumpsuit", and the picture carries the rest.
    - Tested: two women whose outfit was only called "outfit 1" next to bare-shouldered headshots came out undressed. Named for what they wear, they stay dressed.

### Outfit cards (a change of clothes)
- **name:** simple, like a label in a store: "red cocktail dress", "yoga outfit", "gym clothes". With a picture, it's all H3 reads about the outfit.
    - Never a list of pieces. Shoes and pants in the name pull close-ups wide.
    - The name sets the color and the type, the picture fills in the details the name doesn't say. A one-strap red dress with silver details came through from the picture alone, named just "red dress".
- **who:** the person who wears it.
- **description:** head to toe, in detail. Only the picture is made from it.

### Naked, topless, a towel
- An outfit card can be a state of dress: "naked", "topless, black jeans", "wrapped in a white towel". H3 gets it as written ("Nova: ..., topless, wearing black jeans"), never "wearing the topless".
- Typing it in the action works too: "Nova stands naked by the window."
- Showrunner never adds it on its own: the headshots stay head and hair, and everyone is dressed from a card.

### ✋ A color for every item
- **Every piece of clothing gets its own color:** "light-wash denim jacket over a plain white t-shirt, black skinny jeans", not "a jacket and jeans". An item with no color gets a random one, different every time.
- **The shade, not just the color:** light, dark, faded, muted, deep, soft, pale. "faded olive", "deep red", "dusty blue", "cream". Plain "blue" drifts.
- **Layers outside first, with "over":** "denim jacket over a white t-shirt".
- **Patterns name their colors:** "floral blouse in pale peach and green".
- **Fabric helps:** knit, linen, silk, denim, leather, wool.

### How it plays
- Everyone starts in their cast card's 👗 outfit.
- To change: pick another on a shot with the **👗** dropdown, or name it in the action ("Nova comes back to the table in the red dress"). They stay in it from that shot on.
- A close-up only gets its own person's pictures, outfit included, so one person's outfit never lands on the person across the table.
- ⚠️ **Mismatched reference:** a cut back to an earlier camera opens on that shot's first frame. If that frame shows other clothes, the Story warns you. Pin a start frame, or give the scene its own place card.

---

## 📍 Places

- **description:** what is there, as things you could point at: "a small French cafe with a marble counter, big arched windows, bentwood chairs, warm lamps".
- **sound:** sounds you can hear, nothing else: "rain on the glass, cups clinking, footsteps".
    - 🛑 Never write who speaks, "nobody speaks" or "people talking". Naming speech makes H3 speak: the same silent shot talked in French and English with it, and was silent without it.
- **room picture:** **empty**. A person in the room picture shows up in the movie.
- **🌐 360 panorama:** the whole room, horizon level, nobody in it. Make a few, keep the best, then set **📷 the wide shot faces** once.
    - Every close-up gets the right part of the room behind it by itself.
    - 💡 The 360 can't tell left from right. Show it (drag the view), don't tell it.
    - Make the panorama at the story's time of day. A daylight 360 makes every angle look like day.

---

## 👜 Props

- A prop card is for an object that must look the same every time (the umbrella, the notebook, the ring). Cups and chairs don't need one.
- ✋ **The action names it with the card's exact words.** Card "the yellow umbrella" + "The yellow umbrella leans against the table." = the shot gets the picture.
- **description** with colors: "a bright mustard yellow umbrella with a curved black wooden handle".
- **Count once:** "Each of them holds one coffee cup." Say "a cup" twice and you may get two.
- **Box it** (📐) in the first wide shot it appears in, or the storyboard can't place it.

---

## 🎞 Shots

### Framing
- **wide shot:** the whole room and everyone. Open every scene with one: it becomes the place's anchor.
    - ✋ **Wide shots are for movement.** A face there is 45 to 75 px tall and H3 draws in 16 px blocks, so faces and mouths garble. When a face or a line matters, use a medium shot, two-shot, over the shoulder or closer (a face needs about 150 px).
- **side view / two-shot:** two people.
- **medium shot:** one person, waist up. **close-up:** face and shoulders. **extreme close-up:** the face.
- **over the shoulder**, **from behind**, **custom** (you write the camera in the action).

### Who it's on
- **everyone:** everyone the action names (or the whole cast when it names nobody but clearly has people).
- **one person:** a close-up or medium shot is that person alone. Only their pictures are sent.
- **nobody:** an empty shot. The place alone: a ship in space, an empty street, a door closing. No faces are sent.
    - 🛑 A "nobody" shot can only speak as a voiceover (pick the speaker's **(V.O.)**).
    - ⚠️ Don't name a cast member in its action: H3 may put them in.

### 🛑 Don't name the other person in a close-up
- "Anika stares at Nova" made H3 cut to Nova halfway through Anika's own close-up, even with only Anika's pictures sent.
- "Anika stares across the table, surprised." kept her the whole shot (face match 0.32 → 0.79). The Story warns you (⚠️).

### Move
- **still** is the safest. One move per shot at most.
- Moves only happen with H3's own words and a large amount. Small pans barely move. A push-in needs a clear start and a target.

### ✂ Cut or 🎞 keep rolling
- **Cut** for most shots. It gets the faces, the outfits, the props, the room and the anchor.
- **Keep rolling** for the same camera a moment later. It carries on from the last 1.6 s of the shot before, so the join is seamless.
    - 🔒 The card locks the framing, place, angle, outfits and start frame to the shot it rolls from. H3 can't carry on into a new framing: asked to, it cuts by itself to a room it invents.
    - **People can walk in.** Change **who it's on** or name them in the action: they get their face and outfit pictures.
    - 💡 To get closer without a cut, keep rolling with **slow push-in**.

---

## ✍️ Actions

- **Present tense, short sentences, one thing each.** "Nova sits down. She puts her bag on the table. She looks at Anika."
- **Small movements.** Gestures, glances, sitting, standing, a few steps.
- **Say what happens, not what doesn't.** "Nobody drinks" still had her drinking. "Nova rests her hands on the table." works.
- **Name people first, "she" after.**
- **Never invent people.** A waiter with no cast card is a stranger with a random face.
- **Timed beats:** `At 0.00 seconds, Nova looks up. At 1.20 seconds, she smiles.` uses your times exactly.

---

## 🗣️ Lines

- **Word for word** in the **line** box, and pick the **speaker**.
- **One line per shot.** Two people talking = shot and reverse shot.
- **About 2.5 words a second.** Auto length fits it. A typed length that's too short gets stretched.
- **No mouth actions during the line** (drinking, eating, smoking). Write the sip in **after**. "They put their cups down" at the start of a talking shot frees the mouth.
- **after:** what happens once the line is said. Empty = a silent beat, because H3 fills spare time by saying the line again.
- **(V.O.)**: a voiceover, heard off screen with lips closed. The only kind of line an empty shot can have.
- **Don't reorder the cast mid-project.** Speaker 1 is the first person on the cast list for the whole movie.

---

## 🔇 Silent shots

- A shot with no line is silent only if nothing in it mentions speech. The Story adds "Nobody speaks." itself: don't add your own.
- The cast voice clip only goes to shots with a line. A voice clip on a silent shot makes H3 babble.
- No "slightly open mouth" or "about to speak" wording.
- Stop a shot before a person reaches someone they could greet: a walk-up to the counter invited speech.
- Hear a language you didn't write? It's invented speech: look at the place's sound line first.

---

## 🎥 Close-ups across a table

- The Story writes the camera for you: "faces the camera, eyes just past the lens to the left/right, toward the person across the table". That's what makes them look at each other across the cut.
- If you write it yourself: "Seen straight on from across the table, nobody in the foreground: the table edge runs along the bottom of the frame, her body square to the camera, eyes just past the lens to the left."
- Say what's behind each of them in the shot's **behind** box (the 360 angle shows it, the words make it stick).

---

## 🖼 Start frames and 📌 anchors

### Start frames
- A **🖼 start** frame is followed exactly: the shot opens on it.
- ✋ **It has to be the same room as the shot's place.** A start frame of a different café opened the shot, then H3 cut to the place card's café a second later. Using a new room? Make it its own place card.
- ✋ **The start frame is the action's first moment.** Nova seated in the frame + "Nova comes back and sits down" = a second Nova walking in.
- The people in it must look like the cast cards. A stranger in the start frame stays a stranger.
- **🖼 Start shot N here** (in the shot player): any frame of a made shot becomes the next shot's start frame.
- Start frames are for cuts: a keep rolling shot carries on from the shot before.

### Anchors
- A **📌 room anchor** is a frame H3 sees as the room for later cuts in that place: the furniture, the windows, who sits where. It doesn't start a shot.
- The first wide made in a place becomes its anchor by itself. Your own (📌 in the player) replaces it.
- Anchors matter most for shots without a start frame.

### Storyboard boxes (📐)
- One box per person where they stand in the first frame. A prop box takes the prop's name.
- Bigger box = closer. Low in the frame = in front.
- ✋ **A later wide in the same place is built on the first wide's frame, and only the boxes are redrawn.** Box only what changes. Box everyone and everyone is redrawn, and faces drift.
- Check every face on the storyboard before you Make. ↻ the ones that are wrong.

---

## 🔧 When something goes wrong

- **A stranger's face in a shot:** the person has no headshot, or the action names someone with no card.
- **The right face, wrong hair (in every shot):** the look doesn't match the headshot. Rewrite the look to describe the picture.
- **Faces change when people walk into a keep rolling shot:** fixed in LC AV 0.5.17. ↻ the shot.
- **A close-up turns into the other person halfway:** the action names them. Write "across the table" instead.
- **A second copy of someone walks in:** the start frame already shows them where the action has them arriving.
- **Two copies of a person in a close-up:** "who it's on" is "everyone". Set it to one person.
- **People come out undressed:** body words in the look, or an outfit with no real name.
- **Clothes change color between shots:** an item without a color, or a plain one. Give every item a color with a shade.
- **Someone's clothes change every shot:** no 👗 outfit picture.
- **Wrong clothes after a change:** the action didn't use the outfit's exact name, or its **who** is empty.
- **The wrong person talks:** the speaker box, or the cast was reordered after shots were made.
- **The room changed after a cut:** no anchor yet. Make the wide first, or 📌 a good frame from it.
- **A person's face drifts in a later wide:** they were boxed but don't change. Box only what changes.
- **The shot opens on your picture, then jumps to another room:** the start frame isn't the shot's place.
- **Two cups:** counted twice. "Each holds one cup."
- **Garbled speech:** the line is too long for the shot, a mouth action during it, or the face is too small (wide shot).
- **Speech in a silent shot:** speech words in the place's sound line, or a voice clip on it. Sounds only.
- **The line is said twice:** spare time after the line. Add an **after** beat.
- **The cup arrives before the line:** it's in the action, which plays first. Put it in **after**.
- **A person freezes at the end of the shot:** the shot is pinned to an end frame and they got there early. End the action on the move into that frame.
- **Leaving looks like arriving:** both shots use the same angle. Pick another 🌐 angle for the exit.
- **face match under 0.40:** ↻ the shot. If it keeps failing, it's the headshot or the look words.

---

![Lonecat](images/lonecat_logo.png)

*"True Nothing is. Permitted Everything is"*- Yoda Auditore, **Assassin's Wars**
