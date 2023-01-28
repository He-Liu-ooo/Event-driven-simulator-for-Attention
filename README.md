# README
This is an event-driven hardware simulator for simulating the latency of Attention mechanism in ViT under different HW/SW configurations.

## Code structure

    Inheritance relationship:
    BaseUnit ---- SRAM ---- SRAM1
             |         |--- SRAM2
             |
             |--- GlobalBuffer
             |
             |--- CalculatorAndArray
             |
             |--- Softmax

    Hardware structure:
            |--- SRAM1  
    Core ------- SRAM2
            |--- CalculatorAndArray
    GlobalBuffer
    Softmax

                  
