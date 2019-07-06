# DeepInterestNetwork
KDD2018で提案された、CTR予測に関する論文 [Deep Interest Network for Click-Through Rate Prediction](https://arxiv.org/abs/1706.06978) のTensorFlow2.0による実装です。

論文についての解説は、[解説記事](https://pompom168.hatenablog.com/entry/2019/06/30/004050) を参照ください。

# Usage
データセットとして、論文でも使用されていたAmazonのデータセットを動作確認のためサンプリングしたものをリポジトリに同封しています。(dataset_small.pkl)

元のデータセットで使用したい場合は、[こちら](https://github.com/zhougr1993/DeepInterestNetwork)の手順に従ってデータセットを作成して、適宜コピーしてお使いください。

## Training

```
$ python train.py
```
