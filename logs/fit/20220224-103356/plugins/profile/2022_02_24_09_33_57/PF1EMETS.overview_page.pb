?&$	????????I?XWz??HĔH????!?9?ؗl??$	?@a??'@???F؊???d+???@!N??N??@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0{??v????u??&N??A`"ĕ???Y?o?[t??rtrain 81"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?U,~S???<1??PN??A&????}??Y??(????rtrain 82"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0???S????V`??V??A????:??Y?fb????rtrain 83"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0(|????~??k	???A??Ά?3??Y??? ?X??rtrain 84"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0HĔH????9?d??)??A|+????Y????&???rtrain 85"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0u?b?T4?????}V??A?J?4??Y|?????rtrain 86"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?tB?????\???A???|~??Y?HLP÷??rtrain 87"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0	Y??Z?????j{???A?~???Y??Y??Y.???rtrain 88"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0
q?q?t?????Z?[!??A???{,??Y\*?????rtrain 89"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0줾,?T???;?%8??Aj??????Y ?={.??rtrain 90"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?9?ؗl???)?n???Aq? ????YIڍ>???rtrain 91"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails03?ۃ??+??????Aux??q??Y?ŊLÐ?rtrain 92"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0???L??????`??A?r?蜟??Y??r?4??rtrain 93"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0*?t??.??.;??l???A+??-??Y|&??i???rtrain 94"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?/-?????:!t?%??A؛?????Y?m?s???rtrain 95"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0+1?JZ???G?g?u???A??i?{???Y/?:???rtrain 96"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?`?
????G???R{??A${??!U??Y??ׁsF??rtrain 97*	L7?A`A?@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat?? >????!?T+?OA@)?'??Q??1M?C@@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zipk{?????!????_VQ@)%?pt????1~j?G2@:Preprocessing2E
Iterator::Root?u??$???!wp9 ??>@)??&?????1?74???/@:Preprocessing2T
Iterator::Root::ParallelMapV2 ?)U????!F?>!R-@) ?)U????1F?>!R-@:Preprocessing2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice???5??!znp?/ @)???5??1znp?/ @:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate=?ЕT??!?#?Q?)@)??=@????1?j??R@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap??9? ??!|_(??0@)-?????1?P#?|W@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*2V??W??!l???.? @)2V??W??1l???.? @:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 46.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9???kY@I8??4-'X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	??tƛ??U$??:M??.;??l???!???}V??	!       "	!       *	!       2$	??u?ǲ???S?Xê??|+????!q? ????:	!       B	!       J$	F??#?????@ T?w_??fb????!??r?4??R	!       Z$	F??#?????@ T?w_??fb????!??r?4??b	!       JCPU_ONLYY???kY@b q8??4-'X@Y      Y@q9??(?P@"?	
both?Your program is POTENTIALLY input-bound because 46.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?67.4% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 