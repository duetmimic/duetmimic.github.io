// DuetMimic project page: teaser video control, the embedded demo and the
// BibTeX copy button. No dependencies. The demo downloads nothing until the
// visitor presses Start.

(function () {
  'use strict';

  var byId = function (id) { return document.getElementById(id); };
  var demoRunning = false;

  // ------------------------------------------------------------ teaser video
  // Plays muted on a loop, pauses while off screen or while the demo runs, and
  // honours prefers-reduced-motion. The button lets anyone stop the motion.
  var video = byId('teaser-video');
  var toggle = byId('teaser-toggle');
  var userPaused = window.matchMedia &&
    window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  var onScreen = true;

  function showVideoState() {
    if (!toggle) return;
    var paused = video.paused;
    toggle.dataset.state = paused ? 'paused' : 'playing';
    toggle.setAttribute('aria-label', paused ? 'Play video' : 'Pause video');
  }

  function playVideo() {
    var started = video.play();
    // Autoplay can be refused (data saver, low power mode): the button shows Play.
    if (started && started.catch) started.catch(showVideoState);
  }

  // Automatic changes only; a click on the button always wins until the next one.
  function syncVideo() {
    if (!video) return;
    var wanted = !userPaused && onScreen && !demoRunning;
    if (wanted && video.paused) playVideo();
    else if (!wanted && !video.paused) video.pause();
    showVideoState();
  }

  if (video) {
    if (userPaused) video.removeAttribute('autoplay');
    video.addEventListener('play', showVideoState);
    video.addEventListener('pause', showVideoState);
    if (toggle) {
      toggle.addEventListener('click', function () {
        userPaused = !video.paused;
        if (userPaused) video.pause();
        else playVideo();
      });
    }
    if ('IntersectionObserver' in window) {
      new IntersectionObserver(function (entries) {
        onScreen = entries[entries.length - 1].isIntersecting;
        syncVideo();
      }, { threshold: 0.1 }).observe(video);
    }
    syncVideo();
  }

  // ------------------------------------------------------------ demo window
  // On click only: insert the iframe, then hand it keyboard focus so WASD
  // works without a second click. Stop removes it and frees the CPU and GPU.
  var frame = byId('demo-frame');
  var poster = byId('demo-poster');
  var startBtn = byId('demo-start');
  var stopBtn = byId('demo-stop');
  var fullLink = byId('demo-full');
  var iframe = null;

  // The note under Start, before anything downloads: touch screens steer with
  // the demo's on-screen pad, and without WebGPU (navigator.gpu) the networks
  // run on the CPU, in slow motion.
  var hint = byId('demo-hint');
  if (hint) {
    var coarse = window.matchMedia && window.matchMedia('(pointer: coarse)').matches;
    var webgpu = 'gpu' in navigator;
    var parts = [];
    if (coarse) parts.push('steer with the on-screen pad');
    if (!webgpu) {
      parts.push(coarse ? 'slow motion without WebGPU'
        : 'without WebGPU it runs in slow motion (a recent Chrome, Edge or Safari recommended)');
    }
    if (parts.length) hint.textContent = ' · ' + parts.join(' · ');
  }

  function focusDemo() {
    if (!iframe) return;
    iframe.focus({ preventScroll: true });
    try { iframe.contentWindow.focus(); } catch (e) { /* not ready yet */ }
  }

  // Once the page inside has loaded, focus it again, unless the visitor has
  // moved on to something else in the meantime.
  function focusDemoAfterLoad() {
    var active = document.activeElement;
    if (active === iframe || active === document.body || active === null) focusDemo();
  }

  // Bring the started demo fully into view: on laptop-height windows it can
  // start partly below the fold. 'nearest' moves the page as little as
  // possible; the frame's scroll-margin (site.css) keeps the bar below it (key
  // list, Stop, Open full screen) in view when that fits, else the frame's top
  // goes to the top. The CSS scroll-behavior decides smooth or instant
  // (instant under reduced motion).
  function revealDemo() {
    if (!frame.scrollIntoView || !frame.getBoundingClientRect) return;
    var view = window.innerHeight || document.documentElement.clientHeight;
    var box = frame.getBoundingClientRect();
    if (box.top >= 0 && box.bottom <= view) return;
    var style = window.getComputedStyle(frame);
    var margins = (parseFloat(style.scrollMarginTop) || 0) + (parseFloat(style.scrollMarginBottom) || 0);
    frame.scrollIntoView({ block: box.height + margins > view ? 'start' : 'nearest' });
  }

  function startDemo() {
    if (iframe || !frame) return;
    iframe = document.createElement('iframe');
    iframe.className = 'demo-iframe';
    iframe.title = 'DuetMimic interactive demo';
    iframe.setAttribute('allow', 'fullscreen');
    iframe.src = 'demo/?embed=1';
    iframe.addEventListener('load', focusDemoAfterLoad, { once: true });
    frame.appendChild(iframe);
    frame.classList.add('is-live');
    if (poster) poster.hidden = true;
    if (stopBtn) stopBtn.hidden = false;
    demoRunning = true;
    syncVideo();
    focusDemo();
    revealDemo();
  }

  // "Open full screen": while the demo runs, the running iframe goes full
  // screen, so the session goes on; otherwise, without element full screen
  // (iPhone Safari) or if the browser refuses, the link opens demo/ as a page.
  function enterFullscreen(element) {
    var request = element.requestFullscreen || element.webkitRequestFullscreen;
    if (!request || (document.fullscreenEnabled === false && !document.webkitFullscreenEnabled)) return null;
    try {
      var result = request.call(element);
      return result && typeof result.then === 'function' ? result : Promise.resolve();
    } catch (e) {
      return Promise.reject(e);
    }
  }

  if (fullLink) {
    fullLink.addEventListener('click', function (e) {
      if (!iframe || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
      var pending = enterFullscreen(iframe);
      if (!pending) return;
      e.preventDefault();
      pending.then(focusDemo, function () { window.location.href = fullLink.href; });
    });
  }

  function stopDemo() {
    if (!iframe) return;
    iframe.remove();
    iframe = null;
    frame.classList.remove('is-live');
    if (poster) poster.hidden = false;
    if (stopBtn) stopBtn.hidden = true;
    demoRunning = false;
    syncVideo();
    if (startBtn) startBtn.focus({ preventScroll: true });
  }

  if (startBtn) startBtn.addEventListener('click', startDemo);
  if (stopBtn) stopBtn.addEventListener('click', stopDemo);

  // ---------------------------------------------------------------- BibTeX
  var copyBtn = byId('bib-copy');
  var copyLabel = byId('bib-copy-label');
  var bib = byId('bib-text');
  var status = byId('bib-status');
  var copyTimer = 0;

  function copyFallback(text) {
    var area = document.createElement('textarea');
    area.value = text;
    area.setAttribute('readonly', '');
    area.style.position = 'fixed';
    area.style.opacity = '0';
    document.body.appendChild(area);
    area.select();
    var ok = false;
    try { ok = document.execCommand('copy'); } catch (e) { ok = false; }
    area.remove();
    return ok;
  }

  function selectBib() {
    var range = document.createRange();
    range.selectNodeContents(bib);
    var selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
  }

  function reportCopy(ok) {
    copyLabel.textContent = ok ? 'Copied' : 'Selected';
    copyBtn.classList.toggle('is-done', ok);
    if (status) status.textContent = ok ? 'BibTeX copied to the clipboard.' : 'Could not copy. The BibTeX is selected; copy it with the keyboard.';
    if (!ok) selectBib();
    clearTimeout(copyTimer);
    copyTimer = setTimeout(function () {
      copyLabel.textContent = 'Copy';
      copyBtn.classList.remove('is-done');
    }, 1800);
  }

  if (copyBtn && copyLabel && bib) {
    copyBtn.addEventListener('click', function () {
      var text = bib.textContent.replace(/\s+$/, '') + '\n';
      if (navigator.clipboard && window.isSecureContext) {
        navigator.clipboard.writeText(text).then(
          function () { reportCopy(true); },
          function () { reportCopy(copyFallback(text)); });
      } else {
        reportCopy(copyFallback(text));
      }
    });
  }
})();
