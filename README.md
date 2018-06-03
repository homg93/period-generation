# period-generation
Sentence Segmentation to Improve the Translation Accuracy of Automatically Generated Subtitles

Recently, with the development of Speech to Text, which converts voice to text, and machine translation, technologies for simultaneously translating the captions of video into other languages have been developed. Using this, YouTube, a video-sharing site, provides captions in many languages. Currently, the automatic caption system extracts voice data when uploading a video and provides a subtitle file converted into text. This method creates subtitles that have already been created to match the playback time. However, when extracting subtitles from video using Speech to Text, it is impossible to accurately translate the sentence because all connected sentences are generated without a period. So, since the generated subtitles are separated by time units rather than sentence units, and are translated, it is very difficult to understand the translation result as a whole. In this paper, we propose a method to divide text into sentences and generate a period to improve the accuracy of automatic translation of English subtitles. For this study, we use the 10,232 sentence subtitles provided by Stanford University's Natural Language Processing with Deep Learning course as learning data. Since this lecture video provides complete sentence caption data, it can be used as training data by transforming the subtitles into general YouTube caption data. We use LSTM-RNN to train the training data and predict the position of the period. Our research will provide people with more accurate translations of subtitles. In addition, we expect that language barriers in online education will be more easily broken through more accurate translations into numerous video lectures in English.
