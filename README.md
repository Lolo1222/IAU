# Efficient Machine Unlearning via Influence Approximation

This is the accompanying code repository for the paper Efficient Machine Unlearning via Influence Approximation by
**Jiawei Liu**, **Chenwang Wu**, **Defu Lian**, and **Enhong Chen**.

## How to run the experiments

### Data Preparation: 

First, split the CIFAR10 dataset into train/valid set.

```bash
python data_train_valid.py
```

Then divide the parts that need to be forgotten. You can specify the random seed and unlearn ratio.

```bash
python data_remain_unlearn.py --seed=1 --ratio=0.05
```

### Model Training:

#### Traditional Training

Train a CNN model in the full dataset with traditional training to get the initial model. You can change the parameter `dataset ` to remain to get the retrain model, which is the gold model in unlearning. We set the max_epoch, batch_size, and learning_rate to default. You can also specify these value.

```bash
python train_model.py --dataset=origin  --seed=1 --ratio=0.05
# python train_model.py --dataset=origin  --seed=1 --ratio=0.05 --epoch=200 --batch_size=64 --lr=0.001
```

#### Training With Gradient Restriction Loss

The *Gradient Restriction* (GR) loss is

$$
\ell_{GR}(x,\theta)=\ell(z,\theta)+\alpha *||\nabla_{\theta}\ell(z,\theta)||_2
$$

We can use the GR loss by setting parameter `model_fix_flag` to 1. You can specify the value of coefficient $\alpha$.

```bash
python train_model.py --dataset=origin  --seed=1 --ratio=0.05 --model_fix_flag=1 --alpha=0.005
```

### Model Unlearning:

Unlearn on the model with GR loss:

```bash
python IAU_unlearn.py --dataset=origin  --seed=1 --ratio=0.05 --model_fix_flag=1 --alpha=0.005
```

It saves the unlearned model as "ul_model/ul_model.pt".

### Membership Inference Attack:

First we need to train an attack model.

```bash
python generate_mia_data.py
python train_mia.py
```

Then we use this model to attack the unlearned model:

```bash
python mia_attacks.py
```
