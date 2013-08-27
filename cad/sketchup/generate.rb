# This is a Google Sketchup plugin for generated images for various angles

require 'sketchup.rb'

Sketchup.send_action "showRubyPanel:"

UI.menu("Plugins").add_item("Generate data") {

  prompts = ['Name of this model?']
  defaults = ['carN']
  input = UI.inputbox prompts, defaults, 'Give us some info'

  outputname = input[0]

  srand(outputname.hash)

  # Set up rendering style 
  styles = Sketchup.active_model.styles
  
  puts styles["gustav"].class
  style = styles["gustav"]
  if style.instance_of?(Sketchup::Style) then
    styles.selected_style = style
  else
    styles.add_style "/Users/slimgee/git/vision-research/cad/sketchup/gustav.style", true
  end

  model = Sketchup.active_model
  view = model.active_view
  si = 256 

  save_image = Proc.new { |i, altitude, azimuth, out_of_plane, target, dist|
    #rotation = outofplane_index * 6.0 * Math::PI / 180.0
    x = dist * Math.sin(Math::PI / 2.0 - altitude) * Math.cos(azimuth)
    y = dist * Math.sin(Math::PI / 2.0 - altitude) * Math.sin(azimuth) 
    z = dist * Math.cos(Math::PI / 2.0 - altitude) 

    eye = [x + target[0], y + target[1], z + target[2]]
    up= [
      Math.sin(out_of_plane) * Math.cos(azimuth + Math::PI / 2.0), 
      Math.sin(out_of_plane) * Math.sin(azimuth + Math::PI / 2.0), 
      Math.cos(out_of_plane)
    ]
    filename = "/Users/slimgee/git/data/xi3zao3-car2/view%03d_%s.png" % [i, outputname]
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
      true
    else
      false
    end 
  } 

  N = 40 
  #dist = 140 # Bike

  range = (0..N).map { |i| -1 + 2.0 * i/N.to_f } 
  # step(0.1) does not work in SketchUp's Ruby version
  i = 0
  dist = 450 
  dist2 = 300 

  (0..2).each do |altitude_index|
    altitude = ((rand() - 0.5) * 4 + altitude_index * 15.0) * Math::PI / 180.0
    # Do only half
    (0...18).each do |azimuth_index|
      if .. skip front/back
      #[0, -1, 1].each do |outofplane_index|
        #outofplane_index = rand(3) - 1


        # Also change the azimuth angle a bit with the outofplane
        azimuth = (rand() * 4 + azimuth_index * 20.0) * Math::PI / 180.0

        if save_image.call(i, altitude, azimuth, 0.0, [0, 0, 0], dist)
          i += 1
        end
      #end
    end
  end 


  if false
    2.times do |flip|
      6.times do |loop|
        if save_image.call(i, rand() * 7.5 * Math::PI / 180.0, (180 * flip + (rand() - 0.5) * 7.5) * Math::PI / 180.0, (rand() - 0.5) * 2.5 * Math::PI / 180.0, [2.5, 0, 0], dist)
          i += 1
        end
      end
    end
  end

  # Save some specialized frontal shots
  2.times do |flip|
    sgn = [1, -1][flip]
    6.times do |loop|
      if save_image.call(i, rand() * 15 * Math::PI / 180.0, (180.0 * flip + 90.0 + (rand() - 0.5) * 15) * Math::PI / 180.0, (rand() - 0.5) * 5 * Math::PI / 180.0, [0, sgn * 80, 0], dist2)
        i += 1
      end
    end
  end
  UI.messagebox("Data saved! (#{i})")
}

