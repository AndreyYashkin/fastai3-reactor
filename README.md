# fastai3-reactor

This is a draft built on top on fastai2 that shows potential benefits of separate Learners classes for different kinds of problems taking GAN as an example.
`TrainEvalCallback`, `Recorder`, `ProgressCallback` do not work now and some `Learner` functionality is broken.

## Why can't we use one Learner class for all problems?

Current learner with its events is flexible enough to modify, if you want to learn how predict `Y` knowing `X`. However, if you want solve completely different kind of problems `Learner` may appear to be not so extendable. For instance, you may make GANLearner by writing `GANTrainer` callback, but what if someone tries to extend `GANLearner` over existing functionality? As an example to do something with the discriminator input. The only way to do it that I see now is to modify `GANTrainer` itself, but it is not the way to go. To hack `Learner` you can add your callback, but the current `GANLearner` requires modification of its insides. `GANLearner` lacks of events that are related to important steps during the GAN education process. Moreover, the process of solving the problem itself can be built in different ways where key events like `before_crit` will be identical and can be used by callback even with different approaches to solve the minimax problem.

## The vision

I suggest to add following classes: `AbstractLearner`, `AbstractGANLearner`, `GANLearner` and to make `Callback` support a dynamic set of events for different domains.

`AbstractLearner` is a base class for all learners for all kinds of tasks which will be adapted by fastai.
`AbstractGANLearner` is base class for learners designed to educate GAN. It has the functionality which will be shared between like them like sending before\after discrimitator events.
GANLearner is a default learner for GAN.
This will allow to create callbacks with events that are specific for the task you are trying to solve. I added two examples of `AbstractGANLearner` child classes:

`GANLearnerA` is a learner like the current `GANLearner` form fastai2.

`GANLearnerB` is an example of alternative way to do GAN education based on an optimizer developed to solve minimax problems.

Both of these learners share events from `AbstractGANLearner`. With these events you can do some type GAN augmentation and many other things.

## Benefits
- `GANLearner` will support (class) conditional generation without need to any hacks.
- `GANLearner` is now possible and easy to hack with optimized callbacks
