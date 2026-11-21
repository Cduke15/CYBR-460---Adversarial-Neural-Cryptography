# CYBR-460---Adversarial-Neural-Cryptography
Secure Communication with AI assignment by Colby Duke

Instructions and information:

1) Make sure you have Python installed! Run: 'pip install tensorflow numpy matplotlib' in your terminal (These are the libraries required).

2) Download all files and place them into a folder - Open the folder with your IDE.

3) Run 'evaluate_demo.py' - when this runs, it will run some test key(s) that Alice and Bob will share/encrypt/decrypt.
   Eve will attempt to read the key(s) but because I have set noise, and trained them, she will very likely fail with around a 50% chance (each test I've done, she's
   failed).

It will also generate an 'accuracy_curve.png' and 'loss_curve.png' to show it visually along with the terminal reading off the keys, successes, and fails! 

It, by default, will run three tests! Pretty neat!
