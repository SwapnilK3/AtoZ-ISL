from cryptography.fernet import Fernet

# Generate and print a key (do this only once, then store the key securely)
key = Fernet.generate_key()
print("Secret key (store this securely):", key.decode())

# Encrypt labels
labels = """A
B
C
D
E
F
G
H
I
J
K
L
M
N
O
P
Q
R
S
T
U
V
W
X
Y
Z
0
1
2
3
4
5
6
7
8
9"""

f = Fernet(key)
encrypted_labels = f.encrypt(labels.encode('utf-8'))

# Save the encrypted labels to file
with open('model/keypoint_classifier/keypoint_classifier_label.enc', 'wb') as file:
    file.write(encrypted_labels)