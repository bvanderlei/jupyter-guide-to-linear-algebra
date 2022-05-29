# -*- coding: utf-8 -*-
"""
Created on Nov 5 12:07:23 2020

@author: Ben Vanderlei

The purpose of this module is to contain the code that is used for the 
Hill Cipher application in the Jupyter Guide to Linear Algebra.
"""
import numpy as np
import laguide as lag
import random

letter_list =' .?ABCDEFGHIJKLMNOPQRSTUVWXYZ'
alphabet = []
for letter in letter_list:
    alphabet.append(letter)

def AlphaMessage_to_NumericMessage(msg):
    ''' 
    AlphaMessage_to_NumericMessage(msg)
    
    Translates a string to a list of values, based on the indices of the 
    alphabet contained in this module.  Returns a list.  Any characters in
    the string that are not in the alphabet are not included in the list.

    Parameters
    ----------
    msg : String
    
    Returns
    -------
    plaintext: List containing ints
    '''
    plaintext = []
    
    for char in msg.upper():
        if (char in alphabet):
            plaintext.append(alphabet.index(char))
        else:
            print(char,"is not included in the current alphabet.")

    return plaintext


def CheckEncryptionMatrix(A):
    '''
    CheckEncryptionMatrix(A)
    
    Determine if det A has an inverse mod N.  N is the length of the alphabet
    contained in this module.

    Parameters
    ----------
    A: NumPy array object of dimension nxn

    Returns
    -------
    True or False
    '''
    # Check shape of A
    if (A.shape[0] != A.shape[1]):
        print("Encryption matrix must be square.")
        return False

    # Check if det A has inverse mod N
    if (ModularInverse(lag.DeterminantIteration(A),len(alphabet))):
        return True
    else:
        return False


def HillCipherEncryption(msg,A):
    '''
    HillCipherEncryption(msg,A)
    
    Apply Hill Ciper encryption to plaintext msg string using NumPy array A.

    Parameters
    ----------
    msg: String
    A: NumPy array object of dimension nxn

    Returns
    -------
    encrypted_message: String
    '''   
    # Check for valid encryption matrix

    if (A.shape[0] != A.shape[1]):
        print("Encryption not applied.")
        print("Encryption matrix must be square.")
        return msg
    N = A.shape[0]
    
    if (CheckEncryptionMatrix(A) == False):
        print("Encryption not applied.")
        print("Encryption matrix is not compatible with current alphabet.")
        return msg
    
    # Convert to numerical message
    
    plaintext = AlphaMessage_to_NumericMessage(msg)

    # Pad message with random numbers

    while(len(plaintext)%N != 0):
        plaintext.append(random.randint(0,28))

    # Form plaintext array

    P = np.array(plaintext)
    P = P.reshape((int(len(plaintext)/N),N))
    P = P.transpose()

    # Compute ciphertext array
    
    C = (A@P)%len(alphabet)
    C = C.transpose()
    C = C.reshape((1,len(plaintext)))
    
    encrypted_message = NumericMessage_to_AlphaMessage(C)
    return encrypted_message


def HillCipherDecryption(msg,A):
    '''
    HillCipherDecryption(msg,A)
    
    Decodes ciphertext msg generated using Hill Cipher and NumPy array A
    by appling modular inverse of A.

    Parameters
    ----------
    msg: String
    A: NumPy array object of dimension nxn

    Returns
    -------
    decrypted_message: String
    '''
    # Check for valid encryption matrix
    if (A.shape[0] != A.shape[1]):
        print("Encryption not applied.")
        print("Encryption matrix must be square.")
        return msg
    N = A.shape[0]

    if (CheckEncryptionMatrix(A) == False):
        print("Encryption not applied.")
        print("Encryption matrix is not compatible with current alphabet.")
        return msg
    
    # Convert to numerical message
    
    ciphertext = AlphaMessage_to_NumericMessage(msg)

    # Pad message with random numbers (should not be necessary)

    while(len(ciphertext)%N != 0):
        ciphertext.append(random.randint(0,28))

    # Form ciphertext array

    C = np.array(ciphertext)
    C = C.reshape((int(len(ciphertext)/N),N))
    C = C.transpose()

    # Compute plaintext array    

    A_inv = ModularInverseMatrix(A)

    P = (A_inv@C)%len(alphabet)
    P = P.transpose()
    P = P.reshape((1,len(ciphertext)))

    decrypted_message = NumericMessage_to_AlphaMessage(P)
    return decrypted_message


def ModularInverse(a,N):
    '''
    ModularInverse(a,N)
    
    ModularInverse finds the inverse of a, mod N, by direct search.

    Parameters
    ----------
    a: int
    N: int

    Returns
    -------
    i: int
    '''
    for i in range(N):
        if (i*a)%N == 1:
            return i


def ModularInverseMatrix(A):
    '''
    ModularInverseMatrix(A)
    
    ModularInverseMatrix computes the invers of a matrix A inverse mod N,
    with N being the length of the alphabet contained in this module. The 
    inverse matrix is computed with determinant formula and modular inverse 
    of det A.

    Parameters
    ----------
    A: NumPy array object of dimension nxn

    Returns
    -------
    A_inv: NumPy array object of dimension nxn
    '''
    # Check shape of A
    if (A.shape[0] != A.shape[1]):
        print("Inverse matrices only defined for square arrays.")
        return
    
    n = A.shape[0]  # n is number of rows and columns in A
    
    A_inv = np.zeros((n,n),dtype='int')

    det_A = lag.DeterminantIteration(A)
    inv_det_A = ModularInverse(det_A,len(alphabet))

    for i in range(n):
        for j in range(n):
            # Extract the Minor matrix
            Minor = []
            for k in range(n):
                if (k != i):
                    row = []
                    for l in range(n):
                        if (l != j):
                            row.append(A[k,l])
                    Minor.append(row)
            Minor_array = np.array(Minor)

            # Compute the (i,j) cofactor
            cofactor = (-1)**(i+j)*lag.DeterminantIteration(Minor_array)
            A_inv[j,i] = cofactor*inv_det_A

    return A_inv


def NumericMessage_to_AlphaMessage(msg):
    ''' 
    NumericMessage_to_AlphaMessage(msg)
    
    Translates an NumPy array of values into a string, based on the alphabet 
    contained in this module.  Returns a string

    Parameters
    ----------
    msg : NumPy array object of dimension 1XN
    
    Returns
    -------
    D: String
    '''
    N = msg.shape[1]
    
    text = ''

    for i in range(N):
        text = text + alphabet[msg[0,i]%len(alphabet)]

    return text
