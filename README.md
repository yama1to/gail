## Generative Adversarial Imitation Learning (GAIL)
専門家の行動を模倣することで方針を学習する強化学習アルゴリズムである。GAILはGenerative Adversarial Networks (GAN)の枠組みを利用しており、ジェネレータ（エージェントの方針）と識別器が互いに競合する。ジェネレータの目標は専門家の行動を模倣した行動を生成することであり、識別器の目標はエージェントの行動と専門家の行動を区別することである。

### main.py
生成器と識別器は反復して更新される。生成器は専門家の行動と自身の行動の差を最小にすることを目指し、識別器は両者を区別する能力を最大にすることを目指す。時間の経過とともに、生成器の政策は専門家の行動を忠実に模倣する政策に収束していく。


### generator.py
ジェネレータは現在の状態を入力とし、同じ状態におけるエキスパートの行動に似せるべきアクションを生成する。ジェネレータは、エキスパートからの行動であるかのように識別器を欺くことができる行動を生成するように学習する。

### discriminator.py
状態-動作のペアを入力として受け取り、その動作が専門家のものなのか、エージェントのポリシーによるものなのかを判断する。1に近いほど専門家の行動、0に近いほど生成者の行動と判断し、確率値を出力する。

### collect_data.py
エキスパートデータの収集。

### expert_data.py
エキスパートデータの読み込み。



###### git add *; git commit -m 'update ';git push
