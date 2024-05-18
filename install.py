import nltk

# In my case, I had cert problems. Try to download it first without this part
# and if you have problems, uncomment this part and run it again.
# import ssl
# try:
#   _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#   pass
# else:
#   ssl._create_default_https_context = _create_unverified_https_context

nltk.download('cmudict')
nltk.download('averaged_perceptron_tagger')