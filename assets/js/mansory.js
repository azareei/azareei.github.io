$(document).ready(function() {
  // Init Masonry only if grid elements exist
  var $grid = $('.grid');
  if ($grid.length > 0) {
    $grid.masonry({
      gutter: 10,
      horizontalOrder: true,
      itemSelector: '.grid-item',
    });
    // Layout Masonry after each image loads
    $grid.imagesLoaded().progress( function() {
      $grid.masonry('layout');
    });
  }
});
