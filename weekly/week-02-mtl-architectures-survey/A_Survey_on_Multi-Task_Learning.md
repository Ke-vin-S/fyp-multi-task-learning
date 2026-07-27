## A_Survey_on_Multi-Task_Learning.pdf (PDF, 1.2 MB)

- **Title**: A Survey on Multi-Task Learning
- **Created**: D:20221014130927+05'30'
- **Modified**: D:20221106172747-05'00'
- **Pages**: 24
- **Format**: PDF
- **Word Count**: 8402

- Words: 8402 | Chars: 50,065 | Pages: 24

> **Note:** Content was truncated (50,065 of 158,536 chars returned). Use maxChars for a higher limit.

### Content

A Survey on Multi-Task Learning
Yu Zhang,Member, IEEEand Qiang Yang,Fellow, IEEE
Abstract—Multi-Task Learning (MTL) is a learning paradigm in machine learning and its aim is to leverage useful information
contained in multiple related tasks to help improve the generalization performance of all the tasks. In this paper, we give a survey for
MTL from the perspective of algorithmic modeling, applications and theoretical analyses. For algorithmic modeling, we give a definition
of MTL and then classify different MTL algorithms into five categories, including feature learning approach, low-rank approach, task
clustering approach, task relation learning approach and decomposition approach as well as discussing the characteristics of each
approach. In order to improve the performance of learning tasks further, MTL can be combined with other learning paradigms including
semi-supervised learning, active learning, unsupervised learning, reinforcement learning, multi-view learning and graphical models.
When the number of tasks is large or the data dimensionality is high, we review online, parallel and distributed MTL models as well as
dimensionality reduction and feature hashing to reveal their computational and storage advantages. Many real-world applications use
MTL to boost their performance and we review representative works in this paper. Finally, we present theoretical analyses and discuss
several future directions for MTL.
Index Terms—Multi-task learning, machine learning, artificial intelligence
Ç
1INTRODUCTION
H
UMANcan learn multiple tasks simultaneously and dur-
ing this learning process, human can use the knowl-
edge learned in a task to help the learning of another task.
For example, according to our experience in learning to
play tennis and squash, we find that the skill of playing ten-
nis can help learn to play squash and vice versa. Inspired by
such human learning ability, Multi-Task Learning (MTL)
[1], a learning paradigm in machine learning, aims to learn
multiple related tasks jointly so that the knowledge con-
tained in a task can be leveraged by other tasks, with the
hope of improving the generalization performance of all the
tasks at hand.
At its early stage, an important motivation of MTL is to alle-
viate the data sparsity problem where each task has a limited
number of labeled data. In the data sparsity problem, the num-
ber of labeled data in each task is insufficient to train an accu-
rate learner, while MTL aggregates the labeled data in all the
tasks in the spirit of data augmentation to obtain a more accu-
rate learner for each task. From this perspective, MTL can help
reuse existing knowledge and reduce the cost of manual label-
ing for learning tasks. When the era of “big data” comes in
some areas such as computer vision and Natural Language
Processing (NLP), it is found that deep MTL models can
achieve better performance than their single-task counterparts.
One reason that MTL is effective is that it utilizes more data
from different learning tasks when compared with single-task
learning. With more data, MTL can learn more robust and uni-
versal representations for multiple tasks and more powerful
models, leading to better knowledge sharing among tasks, bet-
ter performance of each task and low risk of overfitting in each
task.
MTL is related to other learning paradigms in machine
learning, including transfer learning [2], multi-label learn-
ing [3] and multi-output regression. The setting of MTL is
similar to that of transfer learning but with significant dif-
ferences. In MTL, there is no distinction among different
tasks and the objective is to improve the performance of all
the tasks. However, transfer learning is to improve the per-
formance of a target task with the help of source tasks,
hence the target task plays a more important role than
source tasks. In a word, MTL treats all the tasks equally but
in transfer learning the target task attracts most attentions.
From the perspective of the knowledge flow, flows of
knowledge transfer in transfer learning are from source task
(s) to the target task, but in multi-task learning, there are
flows of knowledge sharing between any pair of tasks,
which is illustrated in Fig. 1a. Continual learning [4], in
which tasks come sequentially, learns tasks one by one,
while MTL is to learn multiple tasks together. In multi-label
learning and multi-output regression, each data point is
associated with multiple labels which can be categorical or
numeric. If we treat each of all the possible labels as a task,
multi-label learning and multi-output regression can be
viewed in some sense as a special case of multi-task learning
where different tasks always share the same data during
both the training and testing phrases. On the one hand,
such characteristic in multi-label learning and multi-output
regression leads to different research issues from MTL. For
example, the ranking loss, which enforces the scores (e.g.,
the classification probability) of labels associated with a
data point to be larger than those of absent labels, can be
Yu Zhang is with the Department of Computer Science and Engineering,
Southern University of Science and Technology, Shenzhen, Guangdong
518055, China, and also with the Peng Cheng Laboratory, Shenzhen,
Guangdong 518066, China. E-mail: yu.zhang.ust@gmail.com.
Qiang Yang is with the Department of Computer Science and Engineering,
Hong Kong University of Science and Technology, Hong Kong.
E-mail: qyang@cse.ust.hk.
Manuscript received 5 Aug. 2019; revised 28 Dec. 2020; accepted 17 Mar. 2021.
Date of publication 31 Mar. 2021; date of current version 7 Nov. 2022.
(Corresponding author: Yu Zhang.)
Recommended for acceptance by L. Chen.
Digital Object Identifier no. 10.1109/TKDE.2021.3070203
5586IEEE TRANSACTIONS ON KNOWLEDGE AND DATA ENGINEERING, VOL. 34, NO. 12, DECEMBER 2022
1041-4347©2021 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission.
See ht_tps://www.ieee.org/publications/rights/index.html for more information.
Authorized licensed use limited to: University of Georgia. Downloaded on June 25,2026 at 02:08:39 UTC from IEEE Xplore.  Restrictions apply.

used for multi-label learning but it does not fit MTL where
different tasks possess different data. On the other hand,
this characteristic in multi-label learning and multi-output
regression is invalid in MTL problems. For example, in a
MTL problem discussed in Section 2.7 where each task is to
predict the disease symptom score of Parkinson for a patient
based on 19 bio-medical features, different patients/tasks
should not share the bio-medical data. In a word, multi-
label learning and multi-output regression are different
from multi-task learning as illustrated in Fig. 1b and hence
we will not survey literature on multi-label learning and
multi-output regression. Moreover, multi-view learning is
another learning paradigm in machine learning, where each
data point is associated with multiple views, each of which
consists of a set of features. Even though different views
have different sets of features, all the views are used
together to learn for the same task and hence multi-view
learning belongs to single-task learning with multiple sets
of features, which is different from MTL as shown in Fig. 1c.
Over past decades, MTL has attracted many attentions in
the artificial intelligence and machine learning communities.
Many MTL models have been devised and many MTL appli-
cations in other areas have been exploited. Moreover, many
analyses have been conducted to study theoretical problems
in MTL. This paper serves as a survey on MTL from the per-
spective of algorithmic modeling, applications and theoretical
analyses. For algorithmic modeling, we first give a definition
for MTL and then classify different MTL algorithms into five
categories: feature learning approach which can be further
categorized into feature transformation and feature selection
approaches, low-rank approach, task clustering approach,
task relation learning approach and decomposition approach.
After that, we discuss the combination of MTL with other
learning paradigms, including semi-supervised learning,
active learning, unsupervised learning, reinforcement learn-
ing, multi-view learning and graphical models. To handle a
large number of tasks, we review online, parallel and distrib-
uted MTL models. For data in a high-dimensional space, fea-
ture selection, dimensionality reduction and feature hashing
are introduced as vital tools to process them. As a promising
learning paradigm, MTL has many applications in various
areas and here we briefly review its applications in computer
vision, bioinformatics, health informatics, speech, NLP, web,
etc. From the perspective of theoretical analyses on MTL, we
review relevant works. At last, we discuss several future
directions for MTL.
1
2  MTL MODELS
In order to fully characterize MTL, we first give the defini-
tion of MTL.
Definition (Multi-Task Learning).Givenmlearning
tasksfT
i
g
m
i¼1
where all the tasks or a subset of them are
related, multi-task learning aims to learn themtasks
together to improve the learning of a model for each task
T
i
by using the knowledge contained in all or some of
other tasks.tu
Based on the definition of MTL, we focus on supervised
learning tasks in this section since most MTL studies fall in
this setting and for other types of tasks, we review them in
the next section. In the setting of supervised learning tasks,
a taskT
i
is usually accompanied by a training datasetD
i
consisting ofn
i
training samples, i.e.,D
i
¼fx
i
j
;y
i
j
g
n
i
j¼1
, where
x
i
j
2R
d
i
is thejth training instance inT
i
andy
i
j
is its label.
We denote byX
i
the training data matrix forT
i
, i.e.,X
i
¼
ðx
i
1
;...;x
i
n
i
Þ. When different tasks lie in the same feature
space implying thatd
i
equalsd
j
for anyi6¼j, this setting is
the homogeneous-feature MTL, and otherwise it corre-
sponds to heterogeneous-feature MTL. Without special
explanation, the default MTL setting is the homogeneous-
feature MTL. Here we need to distinguish the heteroge-
neous-feature MTL from the heterogeneous MTL. In [6], the
heterogeneous MTL is considered to consist of different
types  of  supervised  tasks  including  classification  and
regression problems, and here we generalize it to a more
general setting that the heterogeneous MTL consists of tasks
with different types including supervised learning, unsu-
pervised learning, semi-supervised learning, reinforcement
learning, multi-view learning and graphical models. The
opposite to the heterogeneous MTL is the homogeneous
MTL which consists of tasks with only one type. In a word,
the homogeneous and heterogeneous MTL differ in the type
of learning tasks while the homogeneous-feature MTL is
different from the heterogeneous-feature MTL in terms of
Fig. 1. Illustrations for differences between MTL and other learning paradigms.
1.For an introduction to MTL without technical details, please refer
to [5].
ZHANG AND YANG: SURVEY ON MULTI-TASK LEARNING5587
Authorized licensed use limited to: University of Georgia. Downloaded on June 25,2026 at 02:08:39 UTC from IEEE Xplore.  Restrictions apply.

the original feature representations. Similarly, without spe-
cial explanation, the default MTL setting is the homoge-
neous MTL.
In order to characterize the relatedness in the definition
of MTL, there are three issues to be addressed: when to
share, what to share and how to share.
The ‘when to share’ issue is to make choices between sin-
gle-task and multi-task models for a multi-task problem. Cur-
rently such decision is made by human experts and there are
few works to study it. A simple solution is to formulate such
decision as a model selection problem and then use model
selection techniques, e.g., cross validation, to make decisions,
but this solution is usually computational heavy and may
require much more training data. An advanced solution we
think is to use multi-task models which can degenerate to
their single-task counterparts given some form of model
parameters, for example, problem (33) presented in Section 2.8
which can reduce to multiple single-task models with the
learning of different tasks decoupled when a parameterSS
becomes diagonal. In this case, we can let the training data
determine the form ofSSto make an implicit choice.
‘What to share’ needs to determine the form through
which knowledge sharing among all the tasks could occur.
Usually, there are three forms for ‘what to share’, including
feature, instance and parameter. Feature-based MTL aims
to learn common features among different tasks as a way to
share knowledge. Instance-based MTL identifies useful data
instances in a task for other tasks and then shares knowl-
edge via the identified instances. Parameter-based MTL
uses model parameters (e.g., coefficients in linear models or
weights in deep models) in a task to help learn model
parameters in other tasks in some ways, for example, the
regularization. Existing MTL studies mainly focus on fea-
ture-based and parameter-based methods, and only a few
works belong to the instance-based method. A representa-
tive instance-based method is the multi-task distribution
matching method proposed in [7], which first estimates den-
sity ratios between probabilities that each instance as well as
its label belongs to both its own task and a mixture of all the
tasks and then uses all the weighted training data from all
the tasks based on the estimated density ratios to learn
model parameters for each task. Since the studies on
instance-based MTL are few, we mainly review feature-
based and parameter-based MTL models.
After determining ‘what to share’, ‘how to share’ specifies
concrete ways to share knowledge among tasks. In feature-
based MTL, there is a primary approach: feature learning
approach. The feature learning approach focuses on learning
common feature representations for multiple tasks based on
shallow or deep models, where the learned common feature
representation can be a subset or a transformation of the origi-
nal feature representation. In parameter-based MTL, there are
four main approaches: low-rank approach, task clustering
approach, task relation learning approach and decomposition
approach. The low-rank approach interprets the relatedness
of multiple tasks as the low rankness of the parameter matrix
of these tasks. The task clustering approach is to identify task
clusters, each of which contains similar tasks. The task rela-
tion learning approach aims to learn quantitative relations
between tasks from data automatically. The decomposition
approach decomposes the model parameters of all the tasks
into two or more components, which are penalized by differ-
ent regularizers.
In summary, there are mainly five approaches in the fea-
ture-based and parameter-based MTL. In the following sec-
tions, we review these approaches in a chronological order to
reveal the relations and evolutions among different models.
2.1  Feature Learning Approach
Since tasks are related, it is intuitive to assume that different
tasks share a common feature representation based on the
original features. One reason to learn common feature rep-
resentations instead of directly using the original ones is
that the original representation may not have enough
expressive power for multiple tasks. With the training data
in all the tasks, a more powerful representation can be
learned for all the tasks and this representation can bring
the improvement on the performance.
Based on the relationship between the original feature
representation and the learned one, we can further classify
this category into two sub-categories. The first sub-category
is the feature transformation approach where the learned
representation is a linear or nonlinear transformation of the
original representation and in this approach, each feature in
the learned representation is different from the original fea-
tures. Different from this approach, the feature selection
approach, the second sub-category, selects a subset of the
original features as the learned representation and hence
the learned representation is similar to the original one by
eliminating useless features based on different criteria. In
the following, we introduce these two approaches.
2.1.1  Feature Transformation Approach
The multi-layer feedforwar d neural network [1], which
belongs to the feature transformation approach, is one of
the earliest model for multi-task learning. To see how the
multi-layer feedforward neural network is constructed for
MTL, in Fig. 2 we show an example with an input layer, a
hidden layer and an output layer. The input layer receives
training instances from all the tasks and the output layer
hasmoutput units with one for each task. Here the outputs
of the hidden layer can be viewed as the common feature
representation learned for themtasks and the transforma-
tion from the original representation to the learned one
depends on the weights connecting the input and hidden
layers as well as the activation function adopted in the hid-
den units. Hence, if the activation function in the hidden
Fig. 2. An example for the multi-task feedforward neural network with an
input layer, a hidden layer, and an output layer.
5588IEEE TRANSACTIONS ON KNOWLEDGE AND DATA ENGINEERING, VOL. 34, NO. 12, DECEMBER 2022
Authorized licensed use limited to: University of Georgia. Downloaded on June 25,2026 at 02:08:39 UTC from IEEE Xplore.  Restrictions apply.

layer is linear, then the transformation is a linear function
and otherwise it is nonlinear. Compared with multi-layer
feedforward neural networks used for single-task learning,
the difference in the network architecture lies in the output
layers where in single-task learning, there is only one out-
put unit while in MTL, there aremones. In [8], the radial
basis function network, which has only one hidden layer, is
extended to MTL by greedily determining the structure of
the hidden layer. Different from these neural network mod-
els, Silveret al.[9] propose a context-sensitive multi-task
neural network which has only one output unit shared by
different tasks but has a task-specific context as an addi-
tional input.
Different from multi-layer feedforward neural networks
which are connectionist models, the multi-task feature
learning (MTFL) method [10] is formulated under the regu-
larization framework with the objective function as
min
A;U;b
X
m
i¼1
1
n
i
X
n
i
j¼1
lðy
i
j
;ða
i
Þ
T
U
T
x
i
j
þb
i
ÞþkAk
2
2;1
s:t:UU
T
¼I;
(1)
wherelð;Þdenotes a loss function such as the hinge loss or
square loss,b¼ðb
1
;...;b
m
Þ
T
is a vector of offsets in all the
tasks,U2R
dd
is a square transformation matrix,A2R
dm
contains model parameters of all the tasks with itsith col-
umna
i
as model parameters for theith task after the trans-
formation, the‘
2;1
norm of a matrixAdenoted bykAk
2;1
equals the sum of the‘
2
norm of rows inA,Idenotes an
identity matrix with an appropriate size, andis a positive
regularization parameter. The first term in the objective
function of problem (1) measures the empirical loss on
the training sets of all the tasks and the second one is to
enforceAto be row-sparse via the‘
2;1
norm which is equiv-
alent to selecting features after the transformation, while the
constraint enforcesUto be orthogonal. Different from the
multi-layer feedforward neural network whose hidden rep-
resentations may be redundant, the orthogonality ofUcan
prevent the MTFL method from it. As proved in [10], prob-
lem (1) is equivalent to
min
W;D;b
LðW;bÞþtrðW
T
D
1
WÞs:t:D0;trðDÞ1;
(2)
whereLðW;bÞ¼
P
m
i¼1
1
n
i
P
n
i
j¼1
lðy
i
j
;ðw
i
Þ
T
x
i
j
þb
i
Þdenotes the
total training loss,trðÞdenotes the trace of a square matrix,
w
i
¼Ua
i
is the model parameter forT
i
,W¼ðw
1
;...;w
m
Þ,
0denotes a zero vector or matrix with an appropriate
size,M
1
for any square matrixMdenotes its inverse when
it is nonsingular or otherwise its pseudo inverse, andBC
means thatBCis positive semidefinite. Based on this for-
mulation, we can see that the MTFL method is to learn a fea-
ture covarianceDfor all the tasks, which will be interpreted
in Section 2.8 from a probabilistic perspective. GivenD, the
learning of different tasks can be decoupled and this can
facilitate the parallel computing. When givenW,Dhas an
analytical solution asD¼ðW
T
WÞ
1
2
=trðW
T
WÞ
1
2

and by
plugging this solution into problem (2), we can see that the
regularizer onWis the squared trace norm. Then Argyriou
et al.[11] extend problem (2) to a general formulation where
the second term in the objective function becomestrðW
T
f
ðDÞWÞwithfðDÞoperating on the spectrum ofDand dis-
cuss the condition onfðÞto make the whole problem
convex.
Similar to the MTFL method, the multi-task sparse cod-
ing method [12] is to learn a linear transformation on fea-
tures with the objective function formulated as
min
A;U;b
LðUA;bÞs:t:ka
i
k
1
8i2½m;ku
j
k
2
18j2½D;
(3)
wherea
i
, theith column ofA, contains model parameters of
theith task,u
j
is thejth column inU,½cfor an integerc
denotes a set of integers from 1 toc,kk
1
denotes the‘
1
norm of a vector or matrix and equals the sum of the abso-
lute value of its entries, andkk
2
denotes the‘
2
norm of a
vector. Here the transformationU2R
dD
is also called the
dictionary in sparse coding and shared by all the tasks.
Compared with the MTFL method whereUin problem (1)
is addorthogonal matrix,Uin problem (3) is overcom-
plete, which implies thatDis larger thand, with each col-
umn having a bounded‘
2
norm. Another difference is that
in problem (1)Ais enforced to be row-sparse but in prob-
lem (3) it is only sparse via the first constraint. With a simi-
lar idea to the multi-task sparse coding method, Zhuet al.
[13] propose a multi-task infinite support vector machine
via the Indian buffet process and the difference is that in
[13] the dictionary is sparse and model parameters are non-
sparse. In [14], the spike and slab prior is used to learn
sparse model parameters for multi-output regression prob-
lems where transformed features are induced by Gaussian
processes and shared by different outputs.
Recently deep learning becomes popular due to its capac-
ity to learn nonlinear features, which facilitates the learning
of invariant features for multiple tasks, and hence many
deep multi-task models belonging to this approach have
been proposed with each task modeled by a deep neural
network. Here we classify deep multi-task models in this
approach into three main categories. The first category [15],
[16], [17], [18], [19] is to learn a common feature representa-
tion for multiple tasks by sharing first several layers in a
similar architecture to Fig. 2. However, different from Fig. 2,
deep MTL models in this category have a large number of
shared layers, which have general structures such as convo-
lutional layers and pooling layers. Building on the first cate-
gory, the second category is to use adversarial learning,
which is inspired by generative adversarial networks, to
learn a common feature representation for MTL as did in
[20], [21]. Specifically, there are three networks in such
adversarial multi-task models, including a feature network
N
f
, a classification networkN
c
and a domain networkN
d
.
Based onN
f
,N
c
is to minimize the training loss for all the
tasks, whileN
d
aims to distinguish which task a data
instance is from. The objective function of such models is
usually formulated as
min
u
f
;u
c
max
u
d
X
m
i¼1
1
n
i
X
n
i
j¼1
lðy
i
j
;N
c
ðN
f
ðx
i
j
ÞÞÞl
ce
ðd
i
j
;N
d
ðN
f
ðx
i
j
ÞÞÞ

;
ZHANG AND YANG: SURVEY ON MULTI-TASK LEARNING5589
Authorized licensed use limited to: University of Georgia. Downloaded on June 25,2026 at 02:08:39 UTC from IEEE Xplore.  Restrictions apply.

whereu
f
,u
c
,u
d
denote parameters of three networksN
f
,N
c
,
N
d
, respectively,d
i
j
2f1;...;mgdenotes the task index/
label ofx
i
j
, andl
ce
ð;Þdenotes the cross-entropy loss. Based
on this minimax problem,N
f
is to minimize the training
loss for all the tasks and maximize the cross-entropy loss to
fool the domain network to make the learned feature repre-
sentation indistinguishable to all the tasks. When there is no
domain network, this category can reduce to the first cate-
gory. Moreover, in [21], each task can learn its specific fea-
ture representation to increase the expressive power of the
whole model. The last category is to learn different but
related feature representations for different tasks with the
cross-stitch network [22] as a representative model. Specifi-
cally, given two tasksAandBwith an identical network
architecture,x
i;j
A
(x
i;j
B
) denotes the hidden feature outputted
by thejth unit of theith hidden layer for taskA(B). Then
we can define the cross-stitch operation onx
i;j
A
andx
i;j
B
as
~x
i;j
A
~
x
i;j
B

¼
a
AA
a
AB
a
BA
a
BB

x
i;j
A
x
i;j
B

, where~x
i;j
A
and~x
i;j
B
are new
hidden features after learning the two tasks jointly. When
botha
AB
anda
BA
equal 0, training the two networks jointly
is equivalent to training them independently. The network
architecture of the cross-stitch network is shown in Fig. 3.
Here matrixaa, which is defined asaa
a
AA
a
AB
a
BA
a
BB

, enco-
des feature-level task relations between the two tasks and it
can be learned via the backpropagation method.
2.1.2  Feature Selection Approach
One way to do feature selection in MTL is to use the‘
p;q
norm denoted bykWk
p;q
 kðkw
1
k
p
;...;kw
d
k
p
Þk
q
, wherew
i
denotes theith row ofWandkk
p
denotes the‘
p
norm of a
vector, to achieve the group sparsity. Obozinskiet al.[23]
are among the first to study the multi-task feature selection
(MTFS) problem based on the‘
2;1
norm with the objective
function formulated as
min
W;b
LðW;bÞþkWk
2;1
:(4)
The regularizer onWin problem (4) is to enforceWto be
row-sparse, which in turn helps select important features. In
[23], a path-following algorithm is proposed to solve prob-
lem (4) and then Liuet al.[24] employ an optimal first-order
optimization method to solve it. Compared with problem
(1), we can see that problem (4) is similar to the MTFL
method without learning the transformationU. Leeet al.[25]
propose a weighted‘
2;1
norm for multi-task feature selection
where the weights can be learned as well and problem (4) is
extended in [26] to a general case where feature groups can
overlap with each other. In order to make problem (4) more
robust to outliers, a square-root loss function is investigated
in [27]. Moreover, in order to make speedup, a safe screening
method is proposed in [28] to filter out useless features corre-
sponding to zero rows inWbefore optimizing problem (4).
Liuet al.[29] propose to use the‘
1;1
norm to select features
with the objective function formulated as
min
W;b
LðW;bÞþkWk
1;1
:(5)
A block coordinate descent method is proposed to solve
problem (5). In general, we can use the‘
p;q
norm to select
features for MTL.
In order to attain a more sparse subset of features, Gong
et al.[30] propose a capped-‘
p;1
regularizer for multi-task
feature selection wherep¼1or 2 and the objective function
is formulated as
min
W;b
LðW;bÞþ
X
d
i¼1
minðkw
i
k
p
;uÞ;(6)
wherew
i
denotes theith row ofW. With a given threshold
u, the capped-‘
p;1
regularizer (i.e., the second term in prob-
lem (6)) focuses on rows with smaller‘
p
norms thanu,
which is more likely to be sparse. Whenubecomes large
enough, the capped-‘
p;1
regularizer becomeskWk
p;1
and
hence problem (6) degenerates to problem (4) or (5) whenp
equals 2 or1.
Lozano and Swirszcz [31] propose a multi-level Lasso for
MTL where theðj; iÞth entry in the parameter matrixWis
defined asw
ji
¼u
j
^
w
ji
. Whenu
j
is equal to 0,w
ji
becomes 0
fori2½mand hence thejth feature is not selected by the
model. In this sense,u
j
controls the global sparsity for the
jth feature among themtasks. Moreover, when
^
w
ji
becomes
0,w
ji
is also 0 forionly, implying that thejth feature is not
useful for taskT
i
, and so
^
w
ji
is a local indicator for the spar-
sity in taskT
j
. Based on these observations,u
j
and
^
w
ji
are
expected to be sparse, leading to the objective function for-
mulated as
min
uu;
^
W;b
LðW;bÞþ
1
kuuk
1
þ
2
k
^
Wk
1
s:t:w
ji
¼u
j
^
w
ji
;u
j

0;
(7)
whereuu¼ðu
1
;...;u
d
Þ
T
,
^
W¼ð
^
w
1
;...;
^
w
m
Þ, and the nonneg-
ative constraint onu
j
is to keep the model identifiability. It
has been proved in [31] that problem (7) leads to a regular-
izer
P
d
j¼1
ffiffiffiffiffiffiffiffiffiffiffiffiffi
kw
j
k
1
p
, the square root of the‘
1;
1
2
norm regulari-
zation. Moreover, Wanget al.[32] extend problem (7) to a
general situation where the regularizer becomes
1
P
m
i¼1
k
^
w
i
k
p
p
þ
2
kuuk
q
q
. By utilizing a priori information describing
the task relations in a hierarchical structure, Hanet al.[33]
propose a multi-component product based decomposition
forw
ij
where the number of components in the decomposi-
tion can be arbitrary instead of only 2 in [31], [32]. Similar to
[31], Jebara [34] proposes to learn a binary indicator vector
to do multi-task feature selection based on the maximum
entropy discrimination formalism.
Similar to [33] where a priori information is given to
describe task relations in a hierarchical/tree structure, Kim
and Xing [35] utilize the given tree structure to design a reg-
ularizer onWasfðWÞ¼
P
d
i¼1
P
v2V

v
kw
i;G
v
k
2
, whereV
denotes the set of nodes in the given tree structure,G
v
Fig. 3. The architecture for the cross-stitch network.
5590IEEE TRANSACTIONS ON KNOWLEDGE AND DATA ENGINEERING, VOL. 34, NO. 12, DECEMBER 2022
Authorized licensed use limited to: University of Georgia. Downloaded on June 25,2026 at 02:08:39 UTC from IEEE Xplore.  Restrictions apply.

denotes the set of leaf nodes (i.e., tasks) in a sub-tree rooted
at nodev, andw
i;G
v
denotes a subvector of theith row ofW
indexed byG
v
. This regularizer not only enforces each row
ofWto be sparse as the‘
2;1
norm did in problem (4), but
also induces the sparsity in subsets of each row inWbased
on the tree structure.
Different from conventional multi-task feature selection
methods which assume that different tasks share a set of
original features, Zhouet al.[36] consider a different sce-
nario where useful features in different tasks have no over-
lapping. In order to achieve this, an exclusive Lasso model
is proposed with the objective function formulated as
min
W;b
LðW;bÞþkWk
2
1;2
;
where the regularizer is the squared‘
1;2
norm onW.
Another way to select common features for MTL is to use
sparse priors to design probabilistic or Bayesian models.
For‘
p;1
-regularized multi-task feature selection, Zhanget al.
[37] propose a probabilistic interpretation where the‘
p;1
reg-
ularizer corresponds to a generalized normal prior:w
ji

GNðj0;r
j
;pÞ, wheredenotes a (random) variable when we
do not want to introduce it explicitly. Based on this interpre-
tation, Zhanget al.[37] further propose a probabilistic
framework for multi-task feature selection, in which task
relations and outlier tasks can be identified, based on the
matrix-variate generalized normal prior.
In [38], a generalized horseshoe prior is proposed to do
feature selection for MTL as:
Pðw
i
Þ¼
Z
Y
d
j¼1
Nðw
ji
j0;
u
ji
v
ji
ÞNðu
i
j0;r
2
CÞNðv
i
j0;g
2
CÞdu
i
dv
i
;
whereNðjm;ssÞdenotes a univariate or multivariate nor-
mal distribution withmas the mean andssas the variance
or covariance matrix,u
ji
andv
ji
are thejth entries inu
i
and
v
i
, respectively, andr;gare hyperparameters. HereC
shared by all the tasks denotes the feature correlation matrix
to be learned from data and it encodes an assumption that
different tasks share identical feature correlations. WhenC
becomes an identity matrix which means that features are
independent, this prior degenerates to the horseshoe prior.
Hern

andez-Lobatoet al.[39] propose a probabilistic
model based on the horseshoe prior as
Pðw
ji
Þ¼pðw
ji
Þ
h
ji
d
1h
ji
0
hi
z
j
pðw
ji
Þ
t
ji
d
1t
ji
0
hi
v
i
ð1z
j
Þ
pðw
ji
Þ
g
j
d
1g
j
0
hi
ð1v
i
Þð1z
j
Þ
;
(8)
whered
0
is the probability mass function at zero andpðÞ
denotes the density function of non-zero coefficients. In
Eq. (8),z
j
indicates whether featurejis an outlier (z
j
¼1)or
not (z
j
¼0) andv
i
indicates whether taskT
i
is an outlier
(v
i
¼1) or not (v
i
¼0). Moreover,h
ji
andt
ji
indicate
whether featurejis relevant for the prediction inT
i
(h
ji
;t
ji
¼1) or not (h
ji
;t
ji
¼0), andg
j
indicates whether a
non-outlier featurejis relevant (g
j
¼1) for the prediction
or not (g
j
¼0) in all non-outlier tasks. Based on the above
definitions, the three terms in the right-hand side of Eq. (8)
specify probability  density  functions  ofw
ji
based on
different situations of features and tasks. So this model can
also handle outlier tasks but in a different way from [37].
2.1.3  Comparison Between Two Sub-Categories
The two sub-categories have different characteristics where
the feature transformation approach learns a transformation
of the original features as the new representation but the fea-
ture selection approach selects a subset of the original features
as the new representation for all the tasks. Based on the char-
acteristics of those two approaches, the feature selection
approach can be viewed as a special case of the feature trans-
formation approach when the transformation matrix is a diag-
onal0=1matrix where the diagonal entries with value 1
correspond to the selected features. By selecting a subset of
the original features as the new representation, the feature
selection approach has a better interpretability.
2.2  Low-Rank Approach
The relatedness among multiple tasks can imply the low-rank
ofW, leading to the low-rank approach. For example, if the
ith,jth andkth tasks are related in that the model parameter
w
i
of theith task is a linear combination of those of the other
two tasks, then it is easy to show that the rank ofWis at most
m1and hence of low rank. From this perspective, the more
the relatedness is, the lower the rank ofWis.
Ando and Zhang [40] assume that model parameters of
different tasks share a low-rank subspace in part and specif-
ically,w
i
takes the following form as
w
i
¼u
i
þQQ
T
v
i
:(9)
HereQQ2R
hd
is the shared low-rank subspace by multiple
tasks whereh<d. Then we can write in a matrix form as
W¼UþQQ
T
V. Based on the form ofW, the objective func-
tion proposed in [40] is formulated as
min
U;V;QQ;b
LðUþQQ
T
V;bÞþkUk
2
F
s:t:QQQQ
T
¼I;(10)
wherekk
F
denotes the Frobenius norm. The orthonormal
constraint onQQin problem (10) makes the subspace non-
redundant. Whenis large enough, the optimalUcan
become a zero matrix and hence problem (10) is very similar
to problem (1) except that there is no regularization onVin
problem (10) and thatQQhas a smaller number of rows than
columns. Chenet al.[41] generalize problem (10) as
min
U;V;QQ;b
LðW;bÞþ
1
kUk
2
F
þ
2
kWk
2
F
s:t:W¼UþQQ
T
V;QQQQ
T
¼I:
(11)
When setting
2
to be 0, problem (11) reduces to problem
(10). Even though problem (11) is non-convex, with some
convex relaxation technique, it can be relaxed to the follow-
ing convex problem as
min
W;b;M
LðW;bÞþtr

W
T
MþhI
ðÞ
1
W

s:t:
trðMÞ¼h
0MI
;
(12)
whereh¼
2
=
1
and¼
1
hðhþ1Þ. One advantage of
problem (12) over problem (11) is that the global optimum
of the convex problem (12) is much easier to be obtained
ZHANG AND YANG: SURVEY ON MULTI-TASK LEARNING5591
Authorized licensed use limited to: University of Georgia. Downloaded on June 25,2026 at 02:08:39 UTC from IEEE Xplore.  Restrictions apply.

than that of the non-convex problem (11). Compared with
the alternative objective function (2) in the MTFL method,
problem (12) has a similar formulation whereMmodels the
feature covariance for all the tasks. Problem (10) is extended
in [42] to a general case where differentw
i
’s lie in a mani-
fold instead of a subspace. Moreover, in [43], a latent vari-
able model is proposed forWwith the same decomposition
as Eq. (9) and it can provide a framework for MTL by
modeling more cases than problem (10) such as task cluster-
ing, sharing sparse representation, duplicate tasks and
evolving tasks.
It is well known that using the trace norm as a regular-
izer can make a matrix have low rank and hence this regu-
larization is suitable for MTL. Specifically, an objective
function with the trace norm regularization is proposed in
[44] as
min
W;b
LðW;bÞþkWk
Sð1Þ
;(13)
wherem
i
ðWÞdenotes theith smallest singular value ofW
andkWk
Sð1Þ
¼
P
minðm;dÞ
i¼1
m
i
ðWÞdenotes the trace norm of
matrixW. Based on the trace norm, Han and Zhang [45]
propose a capped trace regularizer with the objective func-
tion formulated as
min
W;b
LðW;bÞþ
X
minðm;dÞ
i¼1
minðm
i
ðWÞ;uÞ:(14)
With the use of the thresholdu, the capped trace regularizer
only penalizes small singular values ofW, which is related
to the determination of the rank ofW. Whenuis large
enough, the capped trace regularizer will become the trace
norm and hence problem (14) will reduce to problem (13).
Moreover, a spectralk-support norm is proposed in [46] as
an improvement over the trace norm regularization.
The trace norm regularization has been extended to regu-
larize model parameters in deep multi-task models. Specifi-
cally, the weights in the last several fully connected layers
of deep multi-task neural networks can be viewed as the
parameters of learners for all the tasks. In this view, the
weights connecting two consecutive layers for one task can
be organized in a matrix and hence the weights of all the
tasks can form a tensor. Based on such tensor representa-
tions, several tensor trace norms, which are based on the
trace norm, are used in [47] as regularizers to identify the
low-rank structure of the parameter tensor.
2.3  Task Clustering Approach
The task clustering approach assumes that different tasks
form several clusters, each of which consists of similar tasks.
As indicated by its name, this approach has a close connec-
tion with clustering algorithms and it can be viewed as an
extension of clustering algorithms to the task level while the
conventional clustering algorithms are on the data level.
Thrun and Sullivan [48] propose the first task clustering
algorithm by using a weighted nearest neighbor classifier for
each task, where the initial weights to define the weighted
euclidean  distance  are  learned  by  minimizing  pairwise
within-class distances and maximizing pairwise between-class
distances simultaneously within each task. Then a task transfer
matrixAis defined with itsði; jÞth entrya
ij
recording the gen-
eralization accuracy obtained for taskT
i
by using taskT
j
’s
distance metric. Based onA,mtasks can be grouped intor
clustersfC
i
g
r
i¼1
by maximizing
P
r
t¼1
1
jC
t
j
P
i;j2C
t
a
ij
,wherejj
denotes the cardinality of a set. After obtaining the cluster
structure among all the tasks, the training data of tasks in a
cluster will be pooled together to learn the final weighted near-
est neighbor classifier. This approach has been extended to an
iterative learning process [49] in a similar way tok-means
clustering.
Bakker and Heskes [50] propose a multi-task Bayesian
neural network model with the network structure similar to
Fig. 2 where input-to-hidden weights are shared by all the
tasks but hidden-to-output weights are task-specific. By
definingw
i
as the vector of hidden-to-output weights for
taskT
i
, the multi-task Bayesian neural network assigns a
mixture of Gaussian prior to it:w
i

P
r
j¼1
p
j
Nðjm
j
;SS
j
Þ,
wherep
j
,m
j
andSS
j
specify the prior, the mean and the
covariance in thejth cluster. For tasks in a cluster, they will
share a Gaussian distribution. Whenrequals 1, this model
degenerates to a case where model parameters of different
tasks share a prior, which is similar to several Bayesian
MTL models such as [51], [52], [53] that are based on Gauss-
ian processes andtprocesses.
Xueet al.[54] deploy the Dirichlet process to do cluster-
ing on task level. Specifically, it defines the prior onw
i
as
w
i
G; GDPða;G
0
Þ8i2½m;
whereDPða;G
0
Þdenotes a Dirichlet process withaas a
positive scaling parameter andG
0
a base distribution. To
see the clustering effect, by integrating outG, the condi-
tional distribution ofw
i
, given model parameters of other
tasksW
i
¼f;w
i1
;w
iþ1
;g,is
Pðw
i
jW
i
;a;G
0
Þ¼
a
m1þa
G
0
þ
1
m1þa
X
m
j¼1;j6¼i
d
w
j
;
whered
w
j
denotes the distribution concentrated at a single
pointw
j
.Sow
i
can be equal to eitherw
j
(j6¼i) with proba-
bility
1
m1þa
, which corresponds to the case that those two
tasks lie in the same cluster, or a new sample fromG
0
with
probability
a
m1þa
, which is the case that taskT
i
forms a
new task cluster. Whenais large, the chance to form a new
task cluster is large and soaaffects the number of task clus-
ters. This model is extended in [55], [56] to a case where dif-
ferent tasks in a task cluster share useful features via a
matrix stick-breaking process and a beta-Bernoulli hierar-
chical prior, respectively, and in [57] where each task is a
compressive sensing task. Moreover, a nested Dirichlet pro-
cess is proposed in [58], [59] to use Dirichlet processes to
learn both task clusters and the state structure of an infinite
hidden Markov model, which handles sequential data in
each task. In [60],w
i
is decomposed asw
i
¼u
i
þQQ
T
i
v
i
simi-
lar to Eq. (9), whereu
i
andQQ
i
are sampled according to a
Dirichlet process.
Different from [50], [54], Jacobet al.[61] aim to learn task
clusters under the regularization framework by considering
three orthogonal aspects, including a global penalty to mea-
sure on average how large the parameters, a measure of
between-cluster variance to quantify the distance among
5592IEEE TRANSACTIONS ON KNOWLEDGE AND DATA ENGINEERING, VOL. 34, NO. 12, DECEMBER 2022
Authorized licensed use limited to: University of Georgia. Downloaded on June 25,2026 at 02:08:39 UTC from IEEE Xplore.  Restrictions apply.

different clusters and a measure of within-cluster variance
to quantify the compactness of task clusters. By combining
these three aspects and adopting some convex relaxation
technique, a convex objective function is formulated as
min
W;b;SS
LðW;bÞþtrðW11
T
W
T
Þþtrð
~
WSS
1
~
W
T
Þ
s:t:
~
W¼WPP;aISSbI;trðSSÞ¼g;
(15)
wherePPdenotes themmcentering matrix,1denotes a
column vector of all ones with its size depending on the
context, anda;b;gare hyperparameters.
Kanget al.[62] extend the MTFL method [10] to the case
with multiple task clusters and aim to minimize the squared
trace norm in each cluster. A diagonal matrix,Q
i
2R
mm
,is
defined as a cluster indicator matrix for theith cluster. The
jth diagonal entry ofQ
i
is equal to 1 if taskT
j
lies in theith
cluster and otherwise 0. Since each task can belong to only
one cluster, it is easy to see that
P
r
i¼1
Q
i
¼I. Based on these
considerations, the objective function is formulated as
min
W;b;fQ
i
g
LðW;bÞþ
X
r
i¼1
kWQ
i
k
2
Sð1Þ
s:t:
Q
i
2f0;1g
mm
P
r
i¼1
Q
i
¼I
:
Whenrequals 1, this method reduces to the MTFL method.
Han and Zhang [63] devise a structurally sparse regular-
izer to cluster tasks with the objective function as
min
W;b
LðW;bÞþ
X
j>i
kw
i
w
j
k
2
:(16)
Problem (16) is a special case of the method proposed in [63]
with only one level of task clusters. The regularizer onW
enforces any pair of columns inWto have a chance to be
identical and after solving problem (16), the cluster struc-
ture can be discovered by comparing columns inW. One
advantage of this structurally sparse regularizer is that the
convex problem (16) can automatically determine the num-
ber of task clusters.
Barzilai and Crammer [64] propose a task clustering
method by definingWasW¼FGwhereF2R
dr
andG2
f0;1g
rm
. With an assumption that each task belongs to
only one cluster, the objective function is formulated as
min
F;G;b
LðFG;bÞþkFk
2
F
s:t:
G2f0;1g
rm
kg
i
k
2
¼18i2½m
;
(17)
whereg
i
denotes theith column ofG. When using the hinge
loss or logistic loss, this non-convex problem can be relaxed
to a min-max problem, which has a global optimum, by uti-
lizing the dual problem with respect toWandband dis-
carding some non-convex constraints.
Zhou and Zhao [65] aim to cluster tasks by identifying
representative tasks which are a subset of the givenmtasks.
If taskT
i
is selected by taskT
j
as a representative task, then
it is expected that model parameters forT
j
are similar to
those ofT
i
.z
ij
is defined as the probability that taskT
j
selects taskT
i
as its representative task. Then based on a
matrixZwhoseði; jÞth entry isz
ij
, the objective function is
formulated as
min
W;b;Z
LðW;bÞþ
1
kWk
2
F
þ
2
X
m
i¼1
X
m
j¼1
z
ij
kw
i
w
j
k
2
2
þ
3
kZk
2;1
s:t:Z
0;Z
T
1¼1:
(18)
The third term in the objective function of problem (18) enfor-
ces the closeness of each pair of tasks based onZand the last
term employs the‘
2;1
norm to enforce the row sparsity ofZ
which implies that the number of representative tasks is lim-
ited. The constraints in problem (18) guarantee that entries in
Zdefine valid probabilities. Problem (18) is related to problem
(16) since the regularizer in problem (16) can be reformulated
as2
P
j>i
kw
i
w
j
k
2
¼min
^
Z
0
P
j>i
^
z
ij
kw
i
w
j
k
2
2
þ
1
^
z
ij

,
where both the regularizer and constraint on
^
Zare different
from those onZin problem (18).
Previous studies assume that each task can belong to
only one task cluster and this assumption seems too restric-
tive. In [66], a GO-MTL method relaxes this assumption by
allowing a task to belong to more than one cluster and
defines a decomposition ofWsimilar to problem (17) as
W¼LSwhereL2R
dr
denotes the latent basis withr<
mandS2R
rm
contains linear combination coefficients for
all the tasks.Sis assumed to be sparse since each task is
generated from only a few columns inLor equivalently
belongs to a small number of clusters. The objective function
is formulated as
min
L;S;b
LðLS;bÞþ
1
kSk
1
þ
2
kLk
2
F
:(19)
Compared with the objective function of multi-task sparse
coding, i.e., problem (3), we can see that when the regulari-
zation parameters take appropriate values, these two prob-
lems are almost equivalent except that in multi-task sparse
coding, the dictionaryUis overcomplete, while here the
number of columns inSis smaller than that of its rows. This
method has been extended in [67] to decompose the param-
eter tensor in the fully connected layers of deep neural
networks.
Among the aforementioned methods, the method in [48]
first identifies the cluster structure and then learns the
model parameters of all the tasks separately, which is not
preferred since the cluster structure learned may be subop-
timal for the model parameters, hence follow-up works
learn model parameters and the cluster structure together.
An important problem in clustering is to determine the
number of clusters and this is also important for this
approach. Out of the above methods, only methods in [54],
[63] can automatically determine the number of task clus-
ters, where the method in [54] depends on the capacity of
the Dirichlet process while the method in [63] relies on the
use of a structurally sparse regularizer. Among all those
models, some belong to Bayesian learning, i.e., [50], [54],
while the rest models are regularized models. Among those
regularized methods, only the objective function proposed
in [63] is convex while others are originally non-convex.
The task clustering approach is related to the low-rank
approach. To see that, suppose that there arertask clusters
(r<m) and all the tasks in a cluster share the same model
parameters, making the parameter matrixWlow-rank with
ZHANG AND YANG: SURVEY ON MULTI-TASK LEARNING5593
Authorized licensed use limited to: University of Georgia. Downloaded on June 25,2026 at 02:08:39 UTC from IEEE Xplore.  Restrictions apply.

the rank at mostr. From the perspective of modeling, by set-
tingu
i
to be a zero vector in Eq. (9), we can see that the decom-
position ofWin [40] becomes similar to those in [64], [66],
which in some sense shows the relation between those two
approaches. Moreover, the equivalence between problems
(12) and (15), two typical methods in the low-rank and task
clustering approaches, has been proved in [68]. The task clus-
tering approach can visualize the learned cluster structure,
which is an advantage over the low-rank approach.
2.4  Task Relation Learning Approach
In MTL, tasks are related and the task relatedness can be
quantitated via task similarity, task correlation, task covari-
ance and so on. Here we use task relations to include all the
quantitative relatedness.
In earlier studies on MTL, task relations are assumed to be
known as a priori information. In [69], [70], each task is
assumed to be similar to any other task and so model parame-
ters of each task will be enforced to approach the average
model parameters of all the tasks. In [71], [72], task similarities
for each pair of tasks are given and these studies utilize the
task similarities to design regularizers to guide the learning of
multiple tasks in a principle that the more similar two tasks
are, the closer the corresponding model parameters are
expected to be. A similar formulation to [71] is proposed in
[73] to estimate the mean of multiple distributions by learning
pairwise task relations and another similar formulation is pro-
posed in [74] for log-density gradient estimation. Given a tree
structure describing relations among tasks in [75], model
parameters of a task corresponding to a node in the tree are
enforced to be similar to those of its parent node.
However, in most applications, task relations are not
available. In this case, learning task relations from data
automatically is a good option. Bonillaet al.[76] propose a
multi-task Gaussian process (MTGP) by defining a prior on
f
i
j
, the functional value forx
i
j
,asfNðj0;SSÞ, wheref¼
ðf
1
1
;...;f
m
n
m
Þ
T
contains the functional values for all the train-
ing data.SS, the cov

[... truncated at 50,000 chars. Total available: 158,536 chars]