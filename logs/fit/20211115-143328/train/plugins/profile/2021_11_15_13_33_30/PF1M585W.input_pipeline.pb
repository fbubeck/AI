$	]S?/?X????E? ??????镲??!гY?????$	r??@ݢJ??@8?9?j? @!V???o.@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??u????0*??D??A??St$???Y?'????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&L?
F%u??z6?>W??A?2ı.n??Y???x?&??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?? ?rh?????<,???AZ??ڊ???Y?q??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&9??m4????0?*???A?):????Y???Q???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&pΈ???????ܵ?|??Ac?ZB>???Y䃞ͪϕ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?ͪ??V??J+???A??b?=??Y????<,??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&@a??+???e??a???A?????Y??j+????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&8gDio?????(\????Aw??/???Yn????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?V?/?'???(\?????AT㥛? ??YjM????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	 o?ŏ?????h o??A??e?c]??YQ?|a2??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
I??&??2??%????Aq???h ??Y?~j?t???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?k	??g???:pΈ??A?=yX???YM??St$??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&:#J{?/????????An4??@???Y ?o_Ι?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&NbX9???????_v??A$???~???Y??3????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?/?$???L?J???Af?c]?F??YV-???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?!?uq??n4??@???A?.n????Y? ?	???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?}8gD??Z??ڊ???A?;Nё\??Y䃞ͪϕ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????Q???J?4??A????߾??Y???????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&/?$?????1w-!??ANbX9???Y46<???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&гY???????????AV-????YQ?|a??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???镲???l??????A?QI??&??Y,e?X??*	     ȍ@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatR'??????!?C??&?A@)??K7?A??1@?1??@@:Preprocessing2F
Iterator::Model??Q????!?D?? @@)?H?}??1????3@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip#??~j???!?];??P@)`??"????1Ѧ?	?d)@:Preprocessing2U
Iterator::Model::ParallelMapV2}??bٽ?!???Xx(@)}??bٽ?1???Xx(@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenatem???{???!??N3?+@)/n????1??x?}?@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSliceX9??v???!?~???@)X9??v???1?~???@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap6?>W[???!??j?Al3@)$???~???1Xx@??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*?
F%u??!b???wR@)?
F%u??1b???wR@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 53.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9]%?
?@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	??j'???&??I????l??????!??????	!       "	!       *	!       2$	87?????5C6/*?????St$???!??b?=??:	!       B	!       J$	?G?????z??R???jM????!?'????R	!       Z$	?G?????z??R???jM????!?'????JCPU_ONLYY]%?
?@b 