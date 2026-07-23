from mimaas import MIMaaSClient
from mimaas.exceptions import ResourceNotFoundError

client = MIMaaSClient(api_url="http://127.0.0.1:5000")


def register():
    # Register a new user (only needed once)
    client.register(
        username="test_user_1",
        email="testuser1@example.com",
        first_name="Sebi",
        surname="Doe3",
        password="Test1234!",
        invite_token="70a119b94c4ab643c5f1f7d230961bb31b4c1cc433c35dbb60f63919fcc6b8fd",
        plan="admin"
    )
   

def login():
    client.login("test_user_1", "Test1234!")

print(register())

print(login())
