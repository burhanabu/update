ACCENTLIVE PRO v2

SHURU KARNA (2 tareeqe)
A) Aasan: folder mein "start.bat" double-click karein (Python 3.10/3.11 chahiye). Pehli baar libraries
   install hongi (10-20 min), phir har baar foran khulta hai.
B) GitHub: saari files + .github/workflows/build.yml repo mein upload -> Actions -> Run workflow ->
   Artifact "AccentLivePro-windows" download karke unzip -> AccentLivePro.exe chalayein.

MODELS: kokoro.onnx aur voices.bin ko "models" folder mein rakh dein. Na hon to app khud download kar legi.

VIRTUAL MIC: app mein "Install virtual microphone" dabayein, allow karein, PC restart.
Windows Sound -> Recording -> "CABLE Output" ko Default Device banayein, "Listen to this device" band rakhein.

VOICE: Adam, Jessica, Sarah. Apni voice: "Clone my voice" mein record ya upload karein, naam dein, "Create my voice".
Phir wo list mein Adam/Jessica ki tarah select hoti hai. Pehli baar clone model download hota hai (~130MB).

BUTTONS: ek baar banate hi background mein tayyar hote hain aur "cache" folder mein save ho jate hain,
isliye agli baar software kholte hi foran bolte hain. Bolte waqt button orange ho jata hai.
Keys 1-9 = buttons, Esc = Stop. Text likh kar Enter = bolo, "+ Button" = button bana lo.

Har call ki shuruaat mein "AI generated voice" wali line bolti hai. Sirf apni awaaz clone karein.
