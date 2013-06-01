# This is a Google Sketchup plugin for generated images for various angles

require 'sketchup.rb'

Sketchup.send_action "showRubyPanel:"

UI.menu("Plugins").add_item("Generate data") {

  prompts = ['Name of this model?']
  defaults = ['carN']
  input = UI.inputbox prompts, defaults, 'Give us some info'

  outputname = input[0]

  # Set up rendering style 
  styles = Sketchup.active_model.styles
  
  puts styles["gustav"].class
  style = styles["gustav"]
  if style.instance_of?(Sketchup::Style) then
    styles.selected_style = style
  else
    styles.add_style "/Users/slimgee/git/vision-research/cad/sketchup/gustav.style", true
  end

  si = 128 
  model = Sketchup.active_model
  view = model.active_view
  N = 40 
  #dist = 140 # Bike
  dist = 450 

  range = (0..N).map { |i| -1 + 2.0 * i/N.to_f } 
  # step(0.1) does not work in SketchUp's Ruby version
  i = 0

  (0..2).each do |elevation_index|
    elevation = (90 - (rand() - 0.5) * 7 - elevation_index * 15.0) * Math::PI / 180.0
    # Do only half
    (0...18).each do |azimuth_index|
      #[0, -1, 1].each do |outofplane_index|
        outofplane_index = rand(3) - 1

        rotation = outofplane_index * 6.0 * Math::PI / 180.0

        # Also change the azimuth angle a bit with the outofplane
        azimuth = (rand() * 10 + azimuth_index * 20.0) * Math::PI / 180.0
        x = dist * Math.sin(elevation) * Math.cos(azimuth)
        y = dist * Math.sin(elevation) * Math.sin(azimuth) 
        z = dist * Math.cos(elevation) 

        eye = [x, y, z]
        target = [0, 0, 0]
        up= [
          Math.sin(rotation) * Math.cos(azimuth + Math::PI / 2.0), 
          Math.sin(rotation) * Math.sin(azimuth + Math::PI / 2.0), 
          Math.cos(rotation)
        ]
        filename = "/Users/slimgee/git/data/xi3zao3-car/view%03d_%s.png" % [i, outputname]
        if x != 0 or y != 0 then
          if not File.exists? filename then
            camera = Sketchup::Camera.new eye, target, up
            view.camera=camera
            keys = {
              :filename => filename,
              :width => si,
              :height => si,
              :antialias => true,
              :compression => 0.9,
              :transparent => true,
            }
            status = view.write_image keys 
          end
          i += 1
        end
      #end
    end
  end 
  UI.messagebox("Data saved! (#{i})")
}

