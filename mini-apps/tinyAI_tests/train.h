#pragma once
#include "moving_image.h"

double learn(MovingImage& img, std::size_t max_epochs, std::size_t batchsize, std::size_t neurons, std::size_t ff,
           type_t scale, type_t lr);
