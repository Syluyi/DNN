Set of all the embedding images I got from my network. 

=== NAMING OF THE IMAGES ===
INPUT_LAYER_LABELSET.png

= INPUT =
all = zowel klinkers als meddeklinkers
klk = klinkers
mk = medeklinkers

= LAYER =
in = input
l1 = layer 1
l2 = layer 2
l3 = layer 3

 = LABELSET =
vowel = vowel/consonant
phone = phone
match = correct/incorrect classification
cat = phone catagory
spectro = spectrograms
	
=== LABEL LEGEND ===
List with the meaning of the colors in the different plots.

= LABELS VOWEL =
blue	vowel
orange	consonant

= LABELS MATCH ALL =
gray	=	correctly classified consonant
blue	=	incorrecty classified consonant
orange	=	incorrectly classified vowel
red		=	incorrectly classified vowel 

= LABELS MATCH KLK & MK =
blue	correct
orange	incorrect

= LABELS CAT ALL =
consonants:
	blue	=	plosive
	pink	=	fricative
	red	 	=	nasal
vowels:
	orange		=	long vocal
	light pink	=	short vocal
	light blue	=	diftong
	gray 		=	approximant


= LABELS CAT KLK & MK =
consonants:
	dark blue = plosive
	gray = frictive
	red = sonorant
vowels:
	orange = long vocal
	pink = short vocal
	light pink = sjwa
	light blue = diftong

=== SETTINGS ===
perplexity = 25
learning rate = 10
nr of input variables = 1024 
