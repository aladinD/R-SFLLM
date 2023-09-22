from src.models.components.wireless_module import WirelessModule

def test_wireless_module():
    num_users = 3
    mses = WirelessModule(num_users=num_users).run()
    assert mses.shape == (num_users,)