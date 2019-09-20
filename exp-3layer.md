- resample process of $\theta_2$
- way1: consider $p_2(\theta_2)$, $p_1(\theta_1)$ and $u(\theta_2)$
	- if the model is jointly trained, then 
		- $$f(x)=\int u^*(\theta_2) f_2(x, \theta_2;p_1^*) p_2^*(d\theta_2)$$
	- if we only train the last layer, then 
		- $$f(x)=\int u^\#(\theta_2) f_2(x, \theta_2;p_1^0) p_2^0(d\theta_2)$$
	- Therefore, $p_2^*$ could be estimated by
		- $$p_2^*(\theta_2)=p_2^0(\theta_2) \frac{u^\#(\theta_2)}{u^*(\theta_2)} \frac{f_2(x, \theta_2;p_1^0)}{f_2(x, \theta_2;p_1^*)}$$
	- If we use the fact $u^*(\theta_2)^2=C \|\theta_2\|_{2,p_1^*}^2 + b$, the above equation could be written as
		- $$p_2^*(\theta_2)=p_2^0(\theta_2) \frac{u^\#(\theta_2)}{\sqrt{C \|\theta_2\|_{2,p_1^*}^2 + b}} \frac{\int \theta_2(\theta_1) f(\theta_1,x) p_1^0(d\theta_1)}{\int \theta_2(\theta_1) f(\theta_1,x) p_1^*(d\theta_1)}$$
		- Therefore, I thought that we should first learn $p^*_1$?
- way2: consider sample from $\widetilde{p}_2(\widetilde{\theta}_2)$.
	- if the model is jointly trained, then 
		- $$f(x)=\int \widetilde{u}^*(\widetilde{\theta}_2) f_2(x, \widetilde{\theta}_2;p_1^0) \widetilde{p}_2^*(d\widetilde{\theta}_2)$$
	- if we only train the last layer, then
		- $$f(x)=\int \widetilde{u}^\#(\widetilde{\theta}_2) f_2(x, \widetilde{\theta}_2;p_1^0) p_2^0(d\widetilde{\theta}_2)$$
	- Therefore, $\widetilde{p}_2^*$ could be estimated by
		- $$\widetilde{p}_2^*(\widetilde{\theta}_2)=p_2^0(\widetilde{\theta}_2) \frac{u^\#(\widetilde{\theta}_2)}{u^*(\widetilde{\theta}_2)} \frac{f_2(x, \widetilde{\theta}_2;p_1^0)}{f_2(x, \widetilde{\theta}_2;p_1^0)}$$
	- If we use the fact $\widetilde{u}^*(\widetilde{\theta}_2)^2=C \|\widetilde{\theta}_2\|_{2,p_1^0}^2 + b$, the above equation could be written as
		- $$\widetilde{p}_2^*(\widetilde{\theta}_2)=p_2^0(\widetilde{\theta}_2) \frac{u^\#(\widetilde{\theta}_2)}{\sqrt{C \|\widetilde{\theta}_2\|_{2,p_1^0}^2 + b}} $$
	- problems
		- we could also not be able to calculate $p_2^*(\theta_2)$ from $\widetilde{p}_2^*(\widetilde{\theta}_2)$
		- how to do regression to calculate C and b?



- suppose at iter $t$, we have distribution $p_1^{(t)}(\theta_1)$, $p_2^{(t)}(\theta_2)$ and weight $u^{(t)}(\theta_2)$
	1. step 1: use gradient descent (or l2 least regression) to minimize $u^{(t+1)}(\theta_2)$
	2. step 2: fix function value, find a new $p_2^{(t+1)}(\theta_2)$ and $u^{(t+1)}(\theta_2)$ to minimize the regularization term
		$$\lambda_3 \int u^{(t+1)}(\theta_2) p_2^{(t+1)}(d \theta_2) + \lambda_2 \int \|\theta_2\|_{2,p_1^{(t)}}^2 p_2^{(t+1)}(d \theta_2) $$
		- this correspond to a resample on $p_2^{(t)}$, i.e. $p_2^{(t+1)}(\theta_2)=w(\theta_2) p_2^{(t)}(\theta_2)$.
		- if my calculation is correct, we have 
			- $$w(\theta_2)=\sqrt{\frac{[u^{(t+1)}(\theta_2)]^2\lambda_3}{\|\theta_2\|_{2,p_1^{(t)}}^2\lambda_2}}$$