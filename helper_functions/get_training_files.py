'''
This is a helper function to get the training files and set a name for them
'''

# Added : load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, mode='rt', encoding='utf-8')
    # read all text
    text = file.read().split("\n")
    # close the file
    file.close()
    return text

# Added : save list to file


def save_list(lines, filename):
    # convert lines to a single blob of text
    data = '\n'.join(lines)
    # open file
    file = open(filename, 'w', encoding="utf-8")
    # write text
    file.write(data)
    # close file
    file.close()


current_languages = ['Limbum', 'English']
src_lang, dest_lang = map(str, input("Available languages are : " + ', '.join(i.strip().capitalize() for i in
                                                                              current_languages) + "\n\nEnter the source"
                                                                                                   " language and "
                                                                                                   "destination "
                                                                                                   "separated by a "
                                                                                                   "space : ").split())

path_to_file = "limbum_english_training_dataset.txt"
