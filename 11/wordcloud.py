# import matplotlib
# from tensorflow.contrib.distributions.python.ops.bijectors import inline
#
# from wordcloud import WordCloud
# import matplotlib.pyplot as plt
# # %matplotlib inline
#
#
# text = "coffee phone phone phone phone phone phone phone phone phone cat dog dog"
#
# # Generate a word cloud image
# wordcloud = WordCloud(max_font_size=100).generate(text)
#
# # Display the generated image:
# # the matplotlib way:
#
# fig = plt.figure()
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis('off')
# plt.savefig('wordcloud_ex1.svg')
#
# plt.show()

from konlpy.tag import Komoran
komoran = Komoran()
print(komoran.morphs(u'우왕 코모란도 오픈소스가 되었어요'))
print(komoran.nouns(u'오픈소스에 관심 많은 멋진 개발자님들!'))
print(komoran.pos(u'한글형태소분석기 코모란 테스트 중 입니다.'))

