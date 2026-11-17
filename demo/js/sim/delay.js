// Actuator command delay: port of osl_deploy actuator_delay.ActuatorDelay.
//
// Controllers write the command bus (`ctrl`); write(dataCtrl) runs once per
// physics tick, before mj_step: it appends the bus to a ring of the last
// max(lag) + 1 commands and sets each actuator's data.ctrl to the command `lag`
// ticks older, as training's DelayBuffer does per tick. Default lags are the
// middle of each block's training range (manifest training.actuator_delay):
// muscles 15 ticks, OSL motors 5 ticks for this bundle (demo.json
// sim.*.action.blocks[*].delay_ticks).

export class ActuatorDelay {
  // blocks: [{actuator_ids, delay_ticks}] covering every actuator once.
  constructor(nu, blocks) {
    this.nu = nu;
    this.lag = new Int32Array(nu);
    let max = 0;
    for (const block of blocks) {
      const ticks = block.delay_ticks | 0;
      if (ticks < 0) throw new RangeError('Actuator delays must be nonnegative');
      for (const id of block.actuator_ids) this.lag[id] = ticks;
      max = Math.max(max, ticks);
    }
    this.size = max + 1;
    this.ctrl = new Float64Array(nu); // the command bus
    this.history = new Float64Array(this.size * nu);
    this.head = 0;
  }

  // New episode: the bus and the whole history hold the reset command.
  reset(hold, dataCtrl) {
    this.ctrl.set(hold);
    for (let r = 0; r < this.size; r++) this.history.set(this.ctrl, r * this.nu);
    this.head = 0;
    dataCtrl.set(this.ctrl);
  }

  // Advance one tick: append the bus, apply each actuator's delayed command.
  write(dataCtrl) {
    const { nu, size, lag, history } = this;
    this.head = (this.head + 1) % size;
    history.set(this.ctrl, this.head * nu);
    for (let i = 0; i < nu; i++) {
      const row = (this.head - lag[i] + size) % size;
      dataCtrl[i] = history[row * nu + i];
    }
  }
}
