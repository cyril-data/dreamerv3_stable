**Changer l'Actor et le Critic en simple réseau sans les feature_dim ...? Ou avec "feature_dim": self.fullStateSize**

-  différent de feature_dim de l'encoder ... 
- garder le feature_dim...? 
=> **DONE** herite de `nn.Module` plutôt que `BasePolicy` ou `BaseModel` : `Actor(nn.Module)` `Critic(nn.Module)`

**Forward/predict de Dreamer à faire** 

**Faire le rollout avec l'env en définissant le latent et le rexcurrent en batch = nombre d'env**