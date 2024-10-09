Catherine Yu 
NLP HW3 Viterbi Algorithm for POS Tagging

Program:
- Used WSJ_02-21.pos and WSJ_24.pos as the main training corpus. 
- Handled OOV by categorizing common suffixes, symbols, and substrings.
- Outputs a .pos file in the format '{word}\t{tag}\n', with an empty line after each sentence.

To run:
- Program expects one argument: a .words file with one word per line
- Example: if the file to tag is WSJ_23.words, the command to run is:
	python cjy6445_main_HW3.py WSJ_23.words