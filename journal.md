# Autoencoder

- Improper configurations only allowed the network to learn static images of one of the two backgrounds and nothing else.

- The dataloader returned full frames, which did not allow the randomization to randomize between segments.
  The order the network saw the segments was always the same.

- Had to reduce input dimension from 480x640 to 240x320
  Reason was that the linear layer was getting too many parameters relative to the conv layers.

- Stopped following the paper's recommendations on kernel size. Used Conv2dAuto to get right size.240x320

- The loss was being unfair to smaller objects because it was using mean reduction.
  The network got the zero pixels right all the time, but had no incentive to learn anything other than a blur for the objects.

- TODO: Perceptual loss might be needed to increase sharpness.
