Before moving into details of feature engineering and model architecture.

I want to give a brief ? about loss function and metric.

Since we expect probability  and model it as a classification problem, the loss function would be cross entropy. In the case we have different Binary classification for nudity, violence, ... detection, we could have BCE for each. Or we could consider multi task, Mixture of expert model to have multi class cross entropy. We come back to it when we talk about model

A few shot learner

#### metrics and loss function

For evaluation and metrics. For classification task, we look AUC ROC for models comparison. In our case, we can safely assume that the normal content is much larger than the unsafe ones, which means there could be a data imbalance issue. So we might look pr curve and area under pr curve, for the numerical metrics, we have precision and recall, f1 score, f-beta score and accuracy off-line. We could also have those metrics w.r.t each classes or subclass. Regarding to the online metrics, we could have:

+ online metrics:

  + prevalance: (# of harmful posts we didn't predict / # total posts on the platform). But it fails to measure the impact of exposure. For example, a post that have been viewed by 10 mils people is much worse that 30 post that have only be viewed by 100 people
  + valid repeal: (# of posts restored /# of posted predicted as harmful) false positive rate. measure how strict our model is. it would have a trade off between encouraging the DAU. MAU versus limitation of potential exposure of harmful content
  + proactive rate: measure how powerful of the model to remove harmful content from the pool (# of harmful posts detected by model / # of harmful posts detected by model + repeated by users)
+ offline:
  AUC-ROC, predict, recall, accu, f1, fbeta
  (data imbalance) pr curve # TODO
+ online

  + prevalance: rate of missed out hc
  + harmful impressions: measure the impact and exposure of hc
  + valid repeal: how strict the model is #of hc predicted but revoked / # hc predicted
  + proactive rate: # hc predicted / # (hc predicted + flagged manually)
  + user reports per harmful class: nudity, violence, abuse

Those metric gives us insight what the model should improved in the future development.

##### Moving to the feature engineering section

Given a post, I would like to devide features into three sub cateogires

+ author features

includes

user age, id, gender, city, country, region, language (demographic)

records of being flag flag, report, thumsup, comments, followers, following...

(for pineters) user interest vector (last n saved pins)

+ post features
  ```post ID
  author ID
  author IP TODO
  timestamp TODO
  device one hot or embedding
  links contained: check if the links directed to some unsafe website TODO
  Example: Pins ( Title, Description, Link, Board, Tagged topics, ...)
  content```

+ User-post interactions
  ```User ID
  Post ID
  Interaction type: comment(unstructured), flag(one hot), like(numberical), report(one hot, report type), share(numberical) ... (numbers: numberical scaled) TODO
  Interaction value: "This is disgusting", NA, NA, violence, NA
  
  Timestamp
  ```

+ Unstructured:
  + content (feature extracter: backbone of resNet ViT, Bert, CLIP, InstructBlip, Glip)
  + image features
  + text features:
    + text preprocessing: normalization, tokenization
    + vectorization: convert the preprossed text into a meaningful feature vector
  + We adopt NLP model such as BERT, DistilmBERT, multilingual issue, speed issue (fast inference)
    + TODO unicode BPE

image: decode, resize, and normalize the data
feature extraction: pre-trained model to convert unstructured data to a feature vector. CLIP visual encoder or SimCLR
for videos, VideoMOCO
multimodal features
video features

content and user reactions to the post is important to determined if the content is safe

feature embedding:

+ content (unstructured)
+ User reactions
  For comments, use text encoder like BERT to convert each comment to vector, then aggregate (average) the embedding to get a final embedding
  For numerical like number of shares, scale the number to improver model convergence
+ Author features:
  + The author's past interactions can be useful to determine if the post is harmful or not
  + author violation history: number of violations, total user reports, profane words rate ...
  + demographics features: Age TODO, Gender(one hot), City and country,

For categorical features, we could use one hot.

Or we could bucketize numerical features into categorical features and use one hot.

For categorical features with large number of class, like ID, or region, in model development, we use embedding layer to map it to a dense embedding.  And the embedding we learned from model training will be a representation of the item (user or post)

For unstructured data, we use feature extractors. Specifically,

TODO: we use CV backbone like ViT, ResNet50, to map an input image to an vector

## 404 soft page classification

就是说有一些网页可以点开，但里面是无效内容比如物品已经下架，让你design一个model去detect这些网页
特征：有效内容过少 或者重复内容过多
产品类 out of stock not found  this is awkward, sorry , why not look for xxx some hot items
可能特征：
404 图片或者文字
特殊意思的文字：
page is not found
the page you are looking for doesn't exist
There's nothing here
you reach the wrong door
wrong turn
we are working on it
please return back
please redirect
let's find a better place to go (redirect links)
tripadvisor: this page is on vacation.. and you should be too. Let's get you restarted
looks like this page doesn't fit what you were looking for.
Try one of these on for size!

Remember: there could be still a lots of elements in the soft 404 pages. such as a search bar, a way for users to report a broken link, a contact page or a contact form. Home page and the popular pages are usually included in  the hot blog articles or items when try to retrin the user

web scrapting or parser:
text content
image content

even html page -> detect the rate of meaningful unique content (mc) vs. scripts,css,java.html

Basically, you fetch the URL in question. If you get a hard 404, it's easy: the page is dead. But if it returns 200 OK with a page, then we don't know if it's a good page or a soft 404. So we fetch a known bad URL (the parent directory of the original URL plus some random chars). If that returns a hard 404 then we know the host returns hard 404s on errors, and since the original page fetched okay, we know it must be good.

出现 soft 404 的原因

Out of stock特征 (e commerce)
https://www.mageplaza.com/blog/handle-out-of-stock-product-pages.html
肯定有out of stock 字样且有一定的曝光度 不会是hard 404
可能会推荐其它的item 尤其是 ecommerce
inform your customer about the product's return
offer pre-orders and increased shipping time

可能是由网站的网络服务器、内容管理系统或用户的浏览器出于各种原因生成的。

页面内容单薄
如果页面的内容很少或者几乎没有内容，那么谷歌可能会在后台显示该页面为 soft 404。常见的有空白的分类页、标签页、空的博客页面、产品筛选页面等。

重定向链接不相关
有时候开发人员会将一些准备删除的链接重定向到另一个 URL，但忽略了两个页面的相关性，不相关的重定向也可能会导致 soft 404 的问题。这个问题在电子商务网站上很常见，下架的产品或者分类直接随意地被重定向到了另一个产品或者分类页面。

爬虫不能正常抓取文件
当网站阻止 Google 访问 JavaScript 或 CSS 文件时，有时会出现 soft 404，这些文件主要用于呈现一些交互效果页面。如果爬虫不能访问它们，就可能会导致一个软 404 错误。

页面不存在，但返回 200
某个页面已经被删除，但是可能由于服务器配置错误，将其重定向到了主页或自定义的 URL，并返回了 200 成功状态码，会发生软 404 的情况。

CMS 系统 content management system

data: html, result of other trials ("https://github.com/benhoyt/soft404")
extracted text (cleaned)
extracted images (cleaned)
screenshot (content grounding, object detection, 404, words detection)
VIPS 返回的特征blocks -> 每个block的features (坐标 block大小 字体大小 link数目)
object detection + OCR -> extract text features of each block
注意的点：

+ out of stock 不太能考虑和其它正常子节点比较 考虑到电商很多item 页面都差不多 而且很可能不会出现页面内容过少的情况
+ 404 或许可以与当前网站其它页面比较 页面内容过少 (其它手段)
+ Hard negative samples:

  + 404 related text 不一定意味着 soft page 因为可能它在讨论404 code 或者网页设置 等等 所以一定要有其它的feature. (text size, text length, number of contents, textual content)
  + 404 designer store (text size, text length, number of contents, textual content)
+ Hard positive samples:

  + 404 but without any words about 404 (tripadvis), (lack of content, image, lack of words)
+ model architecture:
+ html page and screenshot:

  + block extractor (VIPS) -> block features and content features
  + object detection and OCR -> image features and text extracted from image
  + text encoder -> text features (text from image + informative text in html)
  + html, site, domain, 子节点
    + We might collect texual information from image OCR or from html specific tags
    + (这里注意一件事情，就是ASCII art https://www.reddit.com/r/adventofcode/comments/zhzg3f/2022_day_10_converting_the_crt_ascii_image_to_a/)
      + Render the text to a blocky pixel array
      + Pre-process the image by upscaling and blurring slightly (Tesseract has a hard time with the blocky letters)
      + Save image and run through Tesseract OCR
      + Output a plain string
    + 网址，URL，域名，IP地址，DNS，域名解析
  + link extractor -> link 2 text

How to use VIPS to get output and construct features:

repeated content rate with parent nodes (if applicable): numerical
pick four blocks sort by size, concatenate features together
block features, content features, image features, OCR features, text features (bert, TFIDF)

向这里面的feature 都可能是某一类别soft 404的主要特征 但都不一定是唯一能决定的特征。如果单一使用任何一类型的特征都有可能导致误判
numerical features -> normalized
unstructured features -> image encoder, text encoder

Binary classification:
Baseline: logistic regression
DNN: 2-3 Linear layers -> sigmoid function
We could consider to add FM module increase feature interactions
We could consider to use
Multitask: different sub categories of soft 404

Data preparation:
collect 404



插件提取html内容发送到服务器，对DOM元素进行分析，通过学习的方法找出标题和正文。

里面有一些机器学习和自然语言处理的算法

比如什么是标题，可以有一些特征，h1,h2等标签，位置处于正上方的可能性较大，可能有weight=bold等highlight，再有就是文本相关性等

作者：黑魔法练习生
链接：https://www.zhihu.com/question/21745035/answer/19250637




Reference:

+ Weapon detection

  + https://zhuanlan.zhihu.com/p/633667700
  + https://medium.com/the-modern-scientist/build-a-weapon-detection-algorithm-using-yolov7-8d1787c93f96
  + https://medium.com/@cloudgeek/detecting-weapons-using-deep-learning-model-7f7b409a250
+ OCR model:

  + https://cloud.tencent.com/developer/article/1560769
  + https://tech.meituan.com/2018/06/29/deep-learning-ocr.html
  + Detecting text-rich objects: OCR or object detection? A case study with stopwatch detection
    + https://www.amazon.science/publications/detecting-text-rich-objects-ocr-or-object-detection-a-case-study-with-stopwatch-detection
  + Deriving image-text document surrogates to optimize cognition
    + https://dl.acm.org/doi/abs/10.1145/1600193.1600212
  + Web Page Segmentation and Informative Content Extraction for Effective Information Retrieval https://www.semanticscholar.org/paper/Web-Page-Segmentation-and-Informative-Content-for-Win-Thwin/813ada753262776b29b80971b9212b73d53e815f

## Video recommendation

https://www.kaggle.com/code/tanmay111999/netflix-preprocessing-techniques-visualizations

show_id : Unique ID for every Movie / Tv Show
type : Identifier - A Movie or TV Show
title : Title of the Movie / Tv Show
director : Director of the Movie
cast : Actors involved in the movie / show
country : Country where the movie / show was produced
date_added : Date it was added on Netflix
release_year : Actual Release year of the move / show
rating : TV Rating of the movie / show
duration : Total Duration - in minutes or number of seasons
listed_in : Genre
description : The summary description
