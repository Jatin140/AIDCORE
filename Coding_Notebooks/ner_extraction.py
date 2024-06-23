from keybert import KeyBERT
import spacy
import nltk
from spacy import displacy

NER = spacy.load("en_core_web_sm")
kw_model = KeyBERT()


docs = ["""
         Supervised learning is the machine learning task of learning a function that
         I LOVE my Amazon 4K Fire Stick!!!  My TV is only 5 years old but already out of date!  I couldn‚Äôt load any new apps but now I do it all on my Fire Stick, and I have only had a brief pause in streaming once, most likely due to my wifi connection.  I like it so much I have purchased a second one for one of my other TVs and I have recommended it to so many friends who were having the same issues I was with their relatively new, yet ‚Äúold‚Äù TV.  Since I already have an echo dot in my room, I just tell Alexa what I want on the TV and she does it!  Definitely one of my best purchases for 2020!!!!
      """,
      """
        I have spent several years carrying an Inreach and it is hard to believe how unreliable this device is. Updating firmware is a nightmare. Syncing, nightmare. User Interface is clunky. Getting a computer to both use the sync/update software and actually see the device plugged in via USB is a nightmare. You can either go with a less robust satellite beacon or go full satellite phone - either is a more reliable alternative.
      """,
      """
        I researched for several weeks for an articulating mount for my 50&#34; flat screen.  The reviews for this were outstanding.  So i put it on my wishlist, and received it for Christmas.<br /><br />-first knock against it, the back plate (that mounts flush to the wall) isn't a single piece.  It is 3 pieces of metal bolted together with hex bolts (hex wrench not included).  Other, higher quality, mounts are a single piece for regidity.<br />- a LOT of assembly required.  assembly isn't a big deal for me, as i'm fairly handy and have no trouble following crappy diagram directions.  but when its holding something this valuable... better directions would be appreciated.<br />- the claim is that this fits up to 65&#34; tvs and can hold 165lbs.  well, i can't comment on the 65&#34; claim, but i can say that the tv bracket was maxed out on my 50&#34; (but i'm assuming all tv's over a certain size have mounting screws in the same location, so their claim may be accurate).. but the 165lb claim, i'm calling bs.  my 50&#34; plasma weighs just under 100lbs (which was why i went with what I understood to be a &#34;beefier&#34; mount).  This system has 2 verticle pieces that mount to the tv, and they attach to a horizontal bar that is part of the wall mount.  When i hung my tv, the horizontal bar immediately began to torque and bent.  In the most vertical setting (since this design has a tilt feature), my tv was tilted down 30 degrees.<br /><br />i'm going to try to return this item (without completely deconstructing it.  and will buy Aeon wall mount instead (assuming Amazon doesn't try to screw me with this falsely advertised crap)
      """,
      """This case came in a timely manner and is very pretty! It is exactly what i wanted and expected."""
"""It has excellent quality at a good price. They don't hurt the ears. They fit perfectly to the head. I recommend it.""",
"""The USB3 port is really nice to have. The default brightness and colors are really good. The price is horrible. I'm not sure why 16:10 is better for an artist than 16:9, but I've not had a problem playing video games at a 16:10 ratio resolution. The monitor includes display port and HDMI. The ability to turn and rotate this display anyway you could want is what makes this monitor special.""",
"""I purchased this to mount my Humminbird Helix 5 upon.  Though the mount is too large for the Helix 5, I was able to drill holes in the plate to mount it regardless.  Other than that, it works great.  It's very sturdy and can handle the bumps of my boat cruising along the water without becoming loose or flopping to the side.""",
"""So far i have been using the cameras for two weeks. Set up was easy peasy, pic and sound quality is great. Nigh vision is awesome. I did not upgrade any of the features to record or backup. I manually will record when I need to. For my purposes I want to be able to pop in and check in on the teens. I think I am going to upgrade to the full time / back up recordings. It is entertaining to just look back and watch the animals while we are away. The motion detector alert works and is loud. The two way talk is not, the sound from the speaker is so low you can barely hear, but I can hear in my house very well. I do recommend this is you are looking for a budget friendly, easy to set up and use camera for your home. I have one set up in LV and my office so every door to my home is within view.""",
"""Never took this product outside in the yet and after a month of use the Bluetooth signal cuts out too much and too easy. Support said to reset them. Reset it 4 times and the bluetooth is still way too weak. Had to revert back to an old pair of earbuds I own that still have a very strong bluetooth signal. Do not buy this. And no apology or response from seller to make this right.""",
"""I have used this product for about a year now, very happy I got it. Works great for my XBOX360, well phone, wireless laptop, and my wife's Kindle Fire HD.""",
"""Great, pure copper, heatsinks!<br /><br />While the days of using them on VGA RAM are likely over, they still allow one to place these on small IC's on a variety of devices.  Personally I have them on my Raspberry Pi's and other SBC (Single Board Computers) to help them dissapate as much heat as they can.  While I don't overclock too much I primarily picked these up for the then, unreleased, Raspberry Pi 3 which had rumored early on (read: Reddit and other hobby/maker forums) to really put out some heat while over-utilized.  While I've never had an issue with mine overheating, I cannot say it wasn't because of these heatsinks; since they have been on all 3 of my devices before they ever saw power.<br /><br />They stick well although some care MUST be taken to get the paper backing off the stick-pad as delicately as possible.  I can imagine someone new or looking for something like this might heavy-hand it and get the sticky-pad dirty or fingerprinted.  This will result in loss of sticking power and I can believe that possibly some lower than normal reviews were due to improper handling and use.  Who knows exactly but the batch I've received have worked well and I'm storing them in a cool, dry place for future use.  I expect no problems.""",
"""Exactly what I wanted. Good quality too""",
"""Love it.  The cable lies flat and the fasteners work well.  I used it to run wired internet to my desktop which is on the opposite side of the living room from my cable outlet (and therefore, modem and router).""",
        ]   

for doc in docs:
    lines = doc
    # print(lines)
    # function to test if something is a noun
    is_noun = lambda pos: pos[:2] == 'NN'
    # do the nlp stuff
    tokenized = nltk.word_tokenize(lines)
    nouns = [word for (word, pos) in nltk.pos_tag(tokenized) if is_noun(pos)] 

    raw_text = " ".join(nouns)    
    keywords = kw_model.extract_keywords(raw_text)
    print(keywords)

# Sample output
# [('alexa', 0.5148), ('tv', 0.3576), ('tvs', 0.3544), ('apps', 0.3102), ('learning', 0.2913)]
# [('firmware', 0.5239), ('syncing', 0.4651), ('sync', 0.465), ('device', 0.4219), ('satellite', 0.4192)]
# [('mounts', 0.3819), ('mount', 0.3748), ('tvs', 0.3665), ('screws', 0.3512), ('tv', 0.3365)]
# [('ears', 0.5914), ('quality', 0.4622), ('head', 0.407), ('price', 0.376), ('case', 0.3529)]
# [('usb3', 0.4801), ('brightness', 0.4669), ('hdmi', 0.3939), ('monitor', 0.3847), ('colors', 0.3522)]
# [('humminbird', 0.3975), ('helix', 0.3925), ('boat', 0.3447), ('mount', 0.2628), ('bumps', 0.254)]
# [('cameras', 0.6434), ('camera', 0.5787), ('recordings', 0.4705), ('detector', 0.438), ('features', 0.4147)]
# [('bluetooth', 0.5777), ('earbuds', 0.3797), ('signal', 0.3669), ('pair', 0.2775), ('cuts', 0.2662)]
# [('kindle', 0.574), ('laptop', 0.5022), ('xbox360', 0.4851), ('hd', 0.3873), ('phone', 0.3347)]
# [('heatsinks', 0.5102), ('overheating', 0.4462), ('copper', 0.3983), ('ram', 0.3862), ('vga', 0.3846)]
# [('quality', 1.0)]
# [('modem', 0.5067), ('router', 0.4445), ('cable', 0.4336), ('fasteners', 0.302), ('outlet', 0.2794)] 