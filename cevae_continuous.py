import torch
from torch import nn

class cevae_continuous(nn.Module):
    """
    Causal Effect Variational Autoencoder (CEVAE) model
    Expected variables:
    - x: features with some continuous and some discrete variables
    - t: treatment variable (binary)
    - y: outcome variable (continuous)
    - z: latent variable (with custom dimension)
    """
    def __init__(self, x_dim, num_con_x, t_dim=1, y_dim=1, z_dim=20):
        super(cevae_continuous, self).__init__()
        self.x_dim = x_dim
        self.num_con_x = num_con_x
        self.num_dis_x = x_dim - num_con_x
        self.t_dim = t_dim
        self.y_dim = y_dim
        self.z_dim = z_dim

        self.__build_encoder()
        self.__build_decoder()

    def __reparam_gaussian(self, mu, var):
        eps = torch.randn_like(var)
        return mu + eps*torch.sqrt(var)

    def __build_encoder(self):
        # p(t|x)
        t_encoder = nn.Sequential(
            nn.Linear(self.x_dim, 200),
            nn.ELU(),
            nn.Linear(200, 1),
            nn.Sigmoid() # not in original code but needed for probability output
        )

        # p(y|x, t)
        y_encoder = nn.Sequential(
            nn.Linear(self.x_dim, 200),
            nn.ELU(),
            nn.Linear(200, 200),
            nn.ELU()
        )
        y_1_mu_encoder = nn.Sequential(
            y_encoder,
            nn.Linear(200, 200),
            nn.ELU(),
            nn.Linear(200, self.y_dim)
        )
        y_0_mu_encoder = nn.Sequential(
            y_encoder,
            nn.Linear(200, 200),
            nn.ELU(),
            nn.Linear(200, self.y_dim)
        )

        # p(z|x, y, t)
        # mu is not bounded but logvar is bounded in (0, inf)
        z_encoder = nn.Sequential(
            nn.Linear(self.x_dim + self.y_dim, 200),
            nn.ELU(),
            nn.Linear(200, 200),
            nn.ELU(),
            nn.Linear(200, 200),
            nn.ELU()
        )
        z_1_encoder = nn.Sequential(
            z_encoder,
            nn.Linear(200, 200),
            nn.ELU()
        )
        z_1_mu_encoder = nn.Sequential(
            z_1_encoder,
            nn.Linear(200, self.z_dim)
        )
        z_1_var_encoder = nn.Sequential(
            z_1_encoder,
            nn.Linear(200, self.z_dim),
            nn.Softplus()
        )

        z_0_encoder = nn.Sequential(
            z_encoder,
            nn.Linear(200, 200),
            nn.ELU()
        )
        z_0_mu_encoder = nn.Sequential(
            z_0_encoder,
            nn.Linear(200, self.z_dim)
        )
        z_0_var_encoder = nn.Sequential(
            z_0_encoder,
            nn.Linear(200, self.z_dim),
            nn.Softplus()
        )

        self.encoder_dict = nn.ModuleDict({
            't_encoder': t_encoder,
            'y_1_mu_encoder': y_1_mu_encoder,
            'y_0_mu_encoder': y_0_mu_encoder,
            'z_1_mu_encoder': z_1_mu_encoder,
            'z_1_var_encoder': z_1_var_encoder,
            'z_0_mu_encoder': z_0_mu_encoder,
            'z_0_var_encoder': z_0_var_encoder
        })
        
    
    def __build_decoder(self):
        # p(t|z)
        t_decoder = nn.Sequential(
            nn.Linear(self.z_dim, 200),
            nn.ELU(),
            nn.Linear(200, self.t_dim),
            nn.Sigmoid() # not in original code but needed for probability output
        )

        # p(x_con|z)
        # mu is not bounded but var is bounded in (0, inf)
        x_decoder = nn.Sequential(
            nn.Linear(self.z_dim, 200),
            nn.ELU(),
            nn.Linear(200, 200),
            nn.ELU()
        )
        x_con_decoder = nn.Sequential(
            x_decoder,
            nn.Linear(200, 200),
            nn.ELU(),
        )
        x_con_mu_decoder = nn.Sequential(
            x_con_decoder,
            nn.Linear(200, self.num_con_x),
        )
        x_con_var_decoder = nn.Sequential(
            x_con_decoder,
            nn.Linear(200, self.num_con_x),
            nn.Softplus()
        )

        # p(x_dis|z)
        x_dis_decoder = nn.Sequential(
            x_decoder,
            nn.Linear(200, 200),
            nn.ELU(),
            nn.Linear(200, self.num_dis_x),
            nn.Sigmoid() # not in original code but needed for probability output
        )

        # p(y|z, t)
        # mu is not bounded and var is fixed to 1
        y_1_mu_decoder = nn.Sequential(
            nn.Linear(self.z_dim, 200),
            nn.ELU(),
            nn.Linear(200, 200),
            nn.ELU(),
            nn.Linear(200, 200),
            nn.ELU(),
            nn.Linear(200, self.y_dim)
        )
        y_0_mu_decoder = nn.Sequential(
            nn.Linear(self.z_dim, 200),
            nn.ELU(),
            nn.Linear(200, 200),
            nn.ELU(),
            nn.Linear(200, 200),
            nn.ELU(),
            nn.Linear(200, self.y_dim)
        )

        self.decoder_dict = nn.ModuleDict({
            't_decoder': t_decoder,
            'x_con_mu_decoder': x_con_mu_decoder,
            'x_con_var_decoder': x_con_var_decoder,
            'x_dis_decoder': x_dis_decoder,
            'y_1_mu_decoder': y_1_mu_decoder,
            'y_0_mu_decoder': y_0_mu_decoder
        })

    def forward(self, batch):
        """
        forward pass with input x alone
        """
        x = batch['x']
        t = batch['t'].unsqueeze(1)
        t_hat_enc = t
        t_hat_dec = t

        # Encoder network
        # p(t|x)
        q_t = self.encoder_dict['t_encoder'](x)
        # q_t_reshaped = torch.cat((q_t, 1 - q_t), dim=1)
        # log_q_t_reshaped = torch.log(q_t_reshaped)
        # t_hat_enc = torch.nn.functional.gumbel_softmax(log_q_t_reshaped, tau=1, hard=True, dim=-1)[:, 0].unsqueeze(1) # reparametrization

        # p(y|x, t)
        y_1_mu_enc = self.encoder_dict['y_1_mu_encoder'](x)
        y_0_mu_enc = self.encoder_dict['y_0_mu_encoder'](x)
        y_mu_enc = y_1_mu_enc * t_hat_enc + y_0_mu_enc * (1 - t_hat_enc) # select y based on predicted t
        # y_hat_enc = self.__reparam_gaussian(y_mu_enc, torch.ones_like(y_mu_enc)) # reparametrization
        y_hat_enc = y_mu_enc

        # p(z|x, y, t)
        x_y_concat = torch.cat([x, y_hat_enc], dim=1)
        z_1_mu = self.encoder_dict['z_1_mu_encoder'](x_y_concat)
        z_1_var = self.encoder_dict['z_1_var_encoder'](x_y_concat)
        z_0_mu = self.encoder_dict['z_0_mu_encoder'](x_y_concat)
        z_0_var = self.encoder_dict['z_0_var_encoder'](x_y_concat)
        z_mu = z_1_mu * t_hat_enc + z_0_mu * (1 - t_hat_enc) # select z mu based on predicted t
        z_var = z_1_var * t_hat_enc + z_0_var * (1 - t_hat_enc) # select z var based on predicted t
        # z = self.__reparam_gaussian(z_mu, z_var) # reparametrization
        z = z_mu

        # Decoder network
        # p(t|z)
        p_t = self.decoder_dict['t_decoder'](z)
        # p_t_reshaped = torch.cat((p_t, 1 - p_t), dim=1)
        # log_p_t_reshaped = torch.log(p_t_reshaped)
        # t_hat_dec = torch.nn.functional.gumbel_softmax(log_p_t_reshaped, tau=1, hard=True, dim=-1)[:, 0].unsqueeze(1) # reparametrization
        
        # p(x_con|z)
        x_con_mu = self.decoder_dict['x_con_mu_decoder'](z)
        x_con_var = self.decoder_dict['x_con_var_decoder'](z)

        # p(x_dis|z)
        p_x_dis = self.decoder_dict['x_dis_decoder'](z)
        
        # p(y|z, t)
        y_1_mu_dec = self.decoder_dict['y_1_mu_decoder'](z)
        y_0_mu_dec = self.decoder_dict['y_0_mu_decoder'](z)
        y_mu_dec = y_1_mu_dec * t_hat_dec + y_0_mu_dec * (1 - t_hat_dec)
        # y_hat_dec = self.__reparam_gaussian(y_mu_dec, torch.ones_like(y_mu_dec)) # reparametrization
        y_hat_dec = y_mu_dec
        
        return q_t, y_mu_enc, z_mu, z_var, p_t, x_con_mu, x_con_var, p_x_dis, y_mu_dec, y_hat_dec
    
    def inference(self, batch):
        """
        predicting with input x and pre-determined t
        """
        x = batch['x']
        t = batch['t'].unsqueeze(1)
        t_hat_enc = t
        t_hat_dec = t

        # Encoder network
        # p(t|z) is voided

        # p(y|x, t)
        y_1_mu_enc = self.encoder_dict['y_1_mu_encoder'](x)
        y_0_mu_enc = self.encoder_dict['y_0_mu_encoder'](x)
        y_mu_enc = y_1_mu_enc * t_hat_enc + y_0_mu_enc * (1 - t_hat_enc) # select y based on predicted t
        # y_hat_enc = self.__reparam_gaussian(y_mu_enc, torch.ones_like(y_mu_enc)) # reparametrization
        y_hat_enc = y_mu_enc
        
        # p(z|x, y, t)
        x_y_concat = torch.cat([x, y_hat_enc], dim=1)
        z_1_mu = self.encoder_dict['z_1_mu_encoder'](x_y_concat)
        z_1_var = self.encoder_dict['z_1_var_encoder'](x_y_concat)
        z_0_mu = self.encoder_dict['z_0_mu_encoder'](x_y_concat)
        z_0_var = self.encoder_dict['z_0_var_encoder'](x_y_concat)
        z_mu = z_1_mu * t_hat_enc + z_0_mu * (1 - t_hat_enc) # select z mu based on predicted t
        z_var = z_1_var * t_hat_enc + z_0_var * (1 - t_hat_enc) # select z var based on predicted t
        # z = self.__reparam_gaussian(z_mu, z_var) # reparametrization
        z = z_mu

        # Decoder network
        # p(t|z) is voided
        
        # p(x_con|z)
        x_con_mu = self.decoder_dict['x_con_mu_decoder'](z)
        x_con_var = self.decoder_dict['x_con_var_decoder'](z)

        # p(x_dis|z)
        p_x_dis = self.decoder_dict['x_dis_decoder'](z)
        
        # p(y|z, t)
        y_1_mu_dec = self.decoder_dict['y_1_mu_decoder'](z)
        y_0_mu_dec = self.decoder_dict['y_0_mu_decoder'](z)
        y_mu_dec = y_1_mu_dec * t_hat_dec + y_0_mu_dec * (1 - t_hat_dec)
        # y_hat_dec = self.__reparam_gaussian(y_mu_dec, torch.ones_like(y_mu_dec)) # reparametrization
        y_hat_dec = y_mu_dec
        
        return y_mu_enc, z_mu, z_var, x_con_mu, x_con_var, p_x_dis, y_mu_dec, y_hat_dec
    
    def cate(self, batch):
        """
        predicting CATE with input x and pre-determined t
        """
        inference_0_sample = batch.copy()
        inference_0_sample['t'] = torch.zeros_like(inference_0_sample['t'])
        inference_0 = self.inference(inference_0_sample)

        inference_1_sample = batch.copy()
        inference_1_sample['t'] = torch.ones_like(inference_1_sample['t'])
        inference_1 = self.inference(batch)

        return inference_1[-1] - inference_0[-1]

    def bern_prob_loss(self, pred_p, true_value):
        prob_loss_1 = torch.log(pred_p + 1e-10)
        prob_loss_0 = torch.log(1 - pred_p + 1e-10)
        return -torch.mean(true_value * prob_loss_1 + (1 - true_value) * prob_loss_0)
    
    def gaus_prob_loss(self, pred_mu, pred_var, true_value):
        dist = torch.distributions.Normal(pred_mu, torch.sqrt(pred_var))
        return -torch.mean(dist.log_prob(true_value))
    
    def kl_divergence(self, pred_mus, pred_vars, prior_mu=0, prior_var=1):
        kl_divergence = (torch.log(prior_var / pred_vars) + 0.5 * (pred_vars + (pred_mus - prior_mu)**2) / prior_var - 1)
        return torch.mean(kl_divergence)
    
    def train_loss(self, output, batch):
        q_t, y_mu_enc, z_mu, z_var, p_t, x_con_mu, x_con_var, p_x_dis, y_mu_dec, y_hat_dec = output
        q_t_loss = 0 #self.bern_prob_loss(q_t, batch['t'])
        q_y_loss = self.gaus_prob_loss(y_mu_enc, torch.ones_like(y_mu_enc), batch['y'])
        z_kl_loss = 0 #self.kl_divergence(z_mu, z_var)
        p_t_loss = 0 #self.bern_prob_loss(p_t, batch['t'])
        p_x_con_loss = 0 #self.gaus_prob_loss(x_con_mu, x_con_var, batch['x'][:, -1*self.num_con_x:])
        p_x_dis_loss = 0 #self.bern_prob_loss(p_x_dis, batch['x'][:, :-1*self.num_con_x])
        p_y_loss = self.gaus_prob_loss(y_mu_dec, torch.ones_like(y_mu_dec), batch['y'])
        loss = q_t_loss + q_y_loss + z_kl_loss + p_t_loss + p_x_con_loss + p_x_dis_loss + p_y_loss
        return loss, q_t_loss, q_y_loss, z_kl_loss, p_t_loss, p_x_con_loss, p_x_dis_loss, p_y_loss
    
    def inference_loss(self, output, batch):
        y_mu_enc, z_mu, z_var, x_con_mu, x_con_var, p_x_dis, y_mu_dec, y_hat_dec = output
        loss = 0
        loss += self.gaus_prob_loss(y_mu_enc, torch.ones_like(y_mu_enc), batch['y'])
        loss += self.kl_divergence(z_mu, z_var)
        loss += self.gaus_prob_loss(x_con_mu, x_con_var, batch['x'][:, -1*self.num_con_x:])
        loss += 0 #self.bern_prob_loss(p_x_dis, batch['x'][:, :-1*self.num_con_x])
        loss += self.gaus_prob_loss(y_mu_dec, torch.ones_like(y_mu_dec), batch['y'])
        return loss