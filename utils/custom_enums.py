from enum import Enum
class TRAK_Type_Enum(str, Enum):
    TRAK: str = "TRAK"
    DTRAK: str = "DTRAK"

class TRAK_Num_Timesteps_Enum(int, Enum):
    ONE: int = 1
    TEN: int = 10
    HUNDRED: int = 100

class Dataset_Type_Enum(str, Enum):
    CIFAR10: str = "cifar10"
    CIFAR2: str = "cifar2"

class Model_Type_Enum(str, Enum):
    FULL: str = "full"
    LORA: str = "lora"


#https://softwareengineering.stackexchange.com/a/409986
def validate_enum(enum, Enum_Class):
    try:
        enum = Enum_Class(enum)
        return enum
    except ValueError:
        # more informative error message
        raise ValueError(f"'{enum} is not a valid {str(Enum_Class)}; possible types: {list(Enum_Class)}")
