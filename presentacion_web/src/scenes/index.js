import { makeParticles } from './particles.js'
import chain from './chain.js'
import pipeline from './pipeline.js'
import unet from './unet.js'
import paths from './paths.js'
import sdedit from './sdedit.js'
import curve from './curve.js'
import results from './results.js'

export const scenes = { noise: makeParticles(true), reform: makeParticles(false), chain, pipeline, unet, paths, curve, results, sdedit }
