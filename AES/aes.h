/*
	AES implementation (Rijndael) based on FIPS:
	for AES-256 bit, (Nb, Nk, Nr) = (4, 8, 14)
*/

namespace encdec {


typedef unsigned char Byte;
typedef unsigned int Word;

constexpr int Nb = 4;
constexpr int Nk = 8;
constexpr int Nr = 14;

void keyExpansion(Byte key[4*Nk], Word w[Nb*(Nr+1)]);

void cipher(Byte in[4*Nb], Byte out[4*Nb], Word w[Nb*(Nr+1)]);

void decipher(Byte in[4*Nb], Byte out[4*Nb], Word w[Nb*(Nr+1)]);

}

