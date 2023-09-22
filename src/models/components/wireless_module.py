import numpy as np
from typing import Optional, Callable
from isac.utils import db2lin

class WirelessModule:
    def __init__(
            self,
            num_users: int = 3,
            num_tx: int = 8,
            num_rx: int = 16,
            num_tx_jammer: int = 64,
            bfra_algo: Optional[Callable] = None,
            rx_algo: Optional[Callable] = None,
            power_jammer_db: float = 30,
            num_subcarriers: int = 64,
            num_symbols: int = 14,
            eta: float = 1.,
            power_constraints_db: Optional[list | np.ndarray] = None,
            num_rbs_per_user: Optional[list | np.ndarray] = None,
            noise_power_db: float = -3.,
            center_aoa: float = 0.,
            aoa_spread: float = 10.,
            aoa_spacing: float = 10.,
            center_aoa_jammer: float = 20.,
            aoa_spread_jammer: float = 5., 
            path_loss: float = 10., 
            num_paths: int = 128,
            fc: float = 2.4e9
        ) -> None:
        """Models the transmission and reception und approximate worst-case MIMO-OFDM MAC adversarial attacks. 

        :param num_users: Number of users in the system, defaults to 3
        :type num_users: int, optional
        :param num_tx: Number of transmit antennas for each user, defaults to 8
        :type num_tx: int, optional
        :param num_rx: Number of receive antennas at legitimate Rx, defaults to 16
        :type num_rx: int, optional
        :param num_tx_jammer: Number of transmit antennas at jammer, defaults to 64
        :type num_tx_jammer: int, optional
        :param bfra_algo: Beamforming, power allocation and scheduling strategy, defaults to None
        :type bfra_algo: Optional[Callable], optional
        :param rx_algo: Receive strategy, defaults to None
        :type rx_algo: Optional[Callable], optional
        :param power_jammer_db: Jamming power in dB, defaults to 30
        :type power_jammer_db: float, optional
        :param num_subcarriers: Number of subcarriers per slot, defaults to 64
        :type num_subcarriers: int, optional
        :param num_symbols: Number of symbols per slot, defaults to 14
        :type num_symbols: int, optional
        :param eta: Resilience hyperparameter, defaults to 1.
        :type eta: float, optional
        :param power_constraints_db: Power constraints for each user, defaults to None.
        If None, then all users will have a power constraint of 5 dBm.
        :type power_constraints_db: Optional[list  |  np.ndarray], optional
        :param num_rbs_per_user: _description_, defaults to None.
        If None, all users will be allocated equal resources.
        :type num_rbs_per_user: Optional[list  |  np.ndarray], optional
        :param noise_power_db: _description_, defaults to -3.
        :type noise_power_db: float, optional
        :param center_aoa: Central angle of legitimate signals, defaults to 0.
        :type center_aoa: float, optional
        :param aoa_spread: Angle spread of legitimate signals, defaults to 10.
        :type aoa_spread: float, optional
        :param aoa_spacing: Angle spacing between legitimate users, defaults to 10.
        :type aoa_spacing: float, optional
        :param center_aoa_jammer: Central direction of jamming signals, defaults to 20.
        :type center_aoa_jammer: float, optional
        :param aoa_spread_jammer: Angle spread for jamming signals, defaults to 5.
        :type aoa_spread_jammer: float, optional
        :param path_loss: Path loss in dB, defaults to 10.
        :type path_loss: float, optional
        :param num_paths: Number of propagation paths, defaults to 128
        :type num_paths: int, optional
        :param fc: Carrier frequency, defaults to 2.4e9
        :type fc: float, optional
        """        
        # general numerology
        self.num_users = num_users
        self.num_tx = num_tx
        self.num_rx = num_rx
        self.num_subcarriers = num_subcarriers
        self.num_symbols = num_symbols
        self.num_rbs = self.num_subcarriers * self.num_symbols
        if power_constraints_db is None:
            self.power_constraints = db2lin(5.) * np.ones((self.num_users,)) 
        else:
            self.power_constraints = db2lin(np.asarray(power_constraints_db))
        
        if num_rbs_per_user is None:
            self.num_rbs_per_user = np.floor(self.num_rbs / self.num_users * np.ones((self.num_users,))).astype(np.uint64)
        
        # jammer settings
        self.num_tx_jammer = num_tx_jammer
        self.power_jammer_db = power_jammer_db
        # tx and rx strategies
        self.bfra_algo = bfra_algo
        self.rx_algo = rx_algo
        self.eta = eta
        # channel settings
        self.noise_power_db = noise_power_db
        self.center_aoa = center_aoa
        self.aoa_spread = aoa_spread
        self.aoa_spacing = aoa_spacing
        self.center_aoa_jammer = center_aoa_jammer
        self.aoa_spread_jammer = aoa_spread_jammer
        self.path_loss = path_loss
        self.num_paths = num_paths
        self.fc = fc

    def run(self) -> np.ndarray:
        """Runs the wireless simulation and returns the communication MSEs for each user.

        :return: Communication MSEs as :class`np.ndarray` with shape (num_users,).
        :rtype: np.ndarray
        """        
        return np.ones((self.num_users)) * 0.1