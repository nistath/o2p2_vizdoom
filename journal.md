# Autoencoder

- Improper configurations only allowed the network to learn static images of one of the two backgrounds and nothing else.

- The dataloader returned full frames, which did not allow the randomization to randomize between segments.
  The order the network saw the segments was always the same.

- Had to reduce input dimension from 480x640 to 240x320
  Reason was that the linear layer was getting too many parameters relative to the conv layers.

- Stopped following the paper's recommendations on kernel size. Used Conv2dAuto to get right size.240x320

- The loss was being unfair to smaller objects because it was using mean reduction.
  The network got the zero pixels right all the time, but had no incentive to learn anything other than a blur for the objects.
  
  - Masking the loss directly ignored getting the background zero pixels right, which was bad for localizing.
  - Some nice results with masks = masks + focus * masks.logical_not()
  - The focus parameter was reinterpreted to mean the focus the loss should place on the
    object vs the background. It was weighted by their relative area.
    
    - one idea is to vary that parameter over the epochs
      first learn how to represent the object and then learn to isolate it 
      It had an interesting result. Should keep it. Wonder what the importance of perceptual loss will be.
    
- TODO: Fix scenario inbalance when blacklisting is not used.

- TODO: Perceptual loss might be needed to increase sharpness.
