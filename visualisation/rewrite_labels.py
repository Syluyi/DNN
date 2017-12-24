# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 12:11:43 2017

@author: Nikki
"""

"""
Code to change the duration_diphthongue row of the metadata file to category information for that input.
"""

replacements = {'duration_diphthongue	phonetext	':'catagory	phonetext',
                '3	p':'pl	p', 
                '3	b':'pl	b', 
                '3	t':'pl	t',
                '3	d':'pl	d',
                '3	k':'pl	k',
                '3	g':'pl	g', 
                '3	f':'fr	f',
                '3	v':'fr	v',
                '3	s':'fr	s',
                '3	z':'fr	z',
                '3	S':'fr	S', #not used
                '3	Z':'fr	Z', #not used
                '3	x':'fr	x',
                '3	G':'fr	G',
                '3	h':'fr	h',
                '3	N':'na	N',
                '3	m':'na	m',
                '3	n':'na	n',
                '3	J':'na	J', #not used
                '3	l':'ap	l',
                '3	r':'ap	r',
                '3	w':'ap	w',
                '3	j':'ap	j',
                '0	j':'ap	j',
                '0	I':'ko	I',
                '0	E':'ko	E',
                '0	A':'ko	A',
                '0	O':'ko	O',
                '0	Y':'ko	Y',
                '0	@':'ko	@',
                '1	i':'la	i',
                '1	y':'la	y',
                '1	e':'la	e',
                '0	0	0	0	0	0	1	1	2':'0	0	0	0	0	0	1	la	2',
                '1	a':'la	a',
                '1	o':'la	o',
                '0	u':'la	u',
                '2	E+':'di	E+',
                '2	Y+':'di	Y+',
                '2	A+':'di	A+'
                }

INFILE = "/ubuntu/data/model/model_F25_L2_fnALL/run_2/embedding_all_v2/meta_old.tsv"
OUTFILE = "/ubuntu/data/model/model_F25_L2_fnALL/run_2/embedding_all_v2/meta.tsv"
with open(INFILE, 'r') as infile, open(OUTFILE, 'w') as outfile:
    for line in infile:
        for src, target in replacements.items():
            line = line.replace(src, target)
        outfile.write(line)