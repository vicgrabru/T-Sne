from mytsnelib import functions

def test_haversine():
    assert functions.haversine(52.370216, 4.895168, 52.520008, 13.404954) == 945793.4375088713